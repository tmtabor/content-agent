"""SQLite connection helpers.

Every call opens a fresh, short-lived connection (data volume here is tiny —
one brand's config plus at most 20 history rows per brand+content-type — so
there's no pooling to be gained). `settings.db_path` is read fresh on every
call rather than cached at import time, so tests can point it at a temp file
via `monkeypatch.setattr(settings, "db_path", ...)`.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from agent.config import settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """Open a connection to the app's SQLite file.

    Foreign keys are off by default in SQLite and must be enabled per
    connection — without this, ON DELETE CASCADE (brand deletion cascading
    to its settings/history rows) silently does nothing.
    """
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Columns added after the initial schema. `CREATE TABLE IF NOT EXISTS` in
# schema.sql only covers a brand-new db file — an existing one (this app has
# no separate migration tool) needs each new column added explicitly here,
# guarded by a PRAGMA table_info check so it's a no-op once already applied.
_ADDITIVE_MIGRATIONS = [
    ("newsletter_settings", "html_template", "TEXT NOT NULL DEFAULT ''"),
]


def _apply_additive_migrations(conn: sqlite3.Connection) -> None:
    for table, column, ddl in _ADDITIVE_MIGRATIONS:
        existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db() -> None:
    """Create the schema if it doesn't exist yet, then apply any additive
    migrations. Safe to call on every startup.
    """
    with get_connection() as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _apply_additive_migrations(conn)
        conn.commit()


@contextmanager
def db_session() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, always close.

    sqlite3.Connection's own context manager only commits/rolls back — it
    never closes the connection, which would leak one file handle per
    repository call. This wraps that with an explicit close.
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
