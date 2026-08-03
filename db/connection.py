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


def _recreate_content_history_with_check(conn: sqlite3.Connection) -> None:
    """SQLite bakes a CHECK constraint's value list into the table at
    creation time — there's no ALTER TABLE to widen it. An existing db's
    content_history table still has whichever CHECK was in effect when it
    was first created; adding a content type (e.g. 'linkedin') means any
    INSERT using it would be rejected by that old CHECK until the table is
    rebuilt. `CREATE TABLE IF NOT EXISTS` alone does NOT fix this — it's a
    no-op against a table that already exists, old CHECK and all.

    A no-op once the current schema.sql's CHECK is already in place (the sql
    substring check below), so safe to call on every startup.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'content_history'"
    ).fetchone()
    if row is None or "'linkedin'" in row["sql"]:
        return  # table doesn't exist yet (fresh db — schema.sql's CREATE TABLE covers it) or already migrated

    conn.executescript(
        """
        CREATE TABLE content_history_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_id TEXT NOT NULL REFERENCES brands (id) ON DELETE CASCADE,
            content_type TEXT NOT NULL CHECK (content_type IN ('bluesky', 'newsletter', 'linkedin')),
            payload TEXT NOT NULL,
            skypilot_post_id TEXT,
            scheduled_for TEXT,
            created_at TEXT NOT NULL
        );
        INSERT INTO content_history_new SELECT * FROM content_history;
        DROP TABLE content_history;
        ALTER TABLE content_history_new RENAME TO content_history;
        CREATE INDEX IF NOT EXISTS idx_history_brand_type
            ON content_history (brand_id, content_type, created_at);
        """
    )


def init_db() -> None:
    """Create the schema if it doesn't exist yet, then apply any migrations.
    Safe to call on every startup.
    """
    with get_connection() as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _apply_additive_migrations(conn)
        _recreate_content_history_with_check(conn)
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
