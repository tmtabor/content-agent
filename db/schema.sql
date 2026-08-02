-- All state for the app lives in one SQLite file (see agent.config.settings.db_path).

CREATE TABLE IF NOT EXISTS brands (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    background TEXT NOT NULL DEFAULT '',
    voice TEXT NOT NULL DEFAULT '',
    audience TEXT NOT NULL DEFAULT '',
    skypilot_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bluesky_settings (
    brand_id TEXT PRIMARY KEY REFERENCES brands (id) ON DELETE CASCADE,
    instructions TEXT NOT NULL DEFAULT '',
    hashtags TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS newsletter_settings (
    brand_id TEXT PRIMARY KEY REFERENCES brands (id) ON DELETE CASCADE,
    instructions TEXT NOT NULL DEFAULT '',
    html_template TEXT NOT NULL DEFAULT ''
);

-- payload shape depends on content_type: BlueskyContent or NewsletterContent
-- (see db/models.py), stored as JSON so future content types need no
-- migration here.
CREATE TABLE IF NOT EXISTS content_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_id TEXT NOT NULL REFERENCES brands (id) ON DELETE CASCADE,
    content_type TEXT NOT NULL CHECK (content_type IN ('bluesky', 'newsletter')),
    payload TEXT NOT NULL,
    skypilot_post_id TEXT,
    scheduled_for TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_brand_type
    ON content_history (brand_id, content_type, created_at);
