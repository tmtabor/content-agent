"""SQLite persistence layer.

A single .db file (see agent.config.settings.db_path) holds all brand
configuration and content history for the app. `init_db()` creates the
schema if it doesn't exist yet; call it once at process startup.
"""

from db.connection import get_connection, init_db

__all__ = ["get_connection", "init_db"]
