"""Unit tests for the SQLite repository layer — brand CRUD, per-content-type
settings, and content history (FIFO trim, purge).
"""

import sqlite3

from agent.config import settings as app_settings
from db.connection import get_connection, init_db
from db.models import BlueskyContent, BlueskyPost, LinkedInContent, NewsletterContent, SubjectPair
from db.repository import (
    add_content_history,
    create_brand,
    delete_brand,
    delete_content_history_entry,
    get_bluesky_settings,
    get_brand,
    get_linkedin_settings,
    get_newsletter_settings,
    list_brands,
    list_content_history,
    purge_content_history,
    recent_bluesky_texts,
    recent_linkedin_texts,
    recent_newsletter_summaries,
    update_bluesky_settings,
    update_brand,
    update_linkedin_settings,
    update_newsletter_settings,
)


def _make_brand(name="Acme"):
    return create_brand(
        name=name, background="A widget company", voice="playful", audience="hobbyists"
    )


def test_create_and_get_brand():
    brand = _make_brand()
    fetched = get_brand(brand.id)
    assert fetched is not None
    assert fetched.name == "Acme"
    assert fetched.background == "A widget company"


def test_get_brand_missing_returns_none():
    assert get_brand("does-not-exist") is None


def test_list_brands_orders_by_creation():
    first = _make_brand("First")
    second = _make_brand("Second")
    assert [b.id for b in list_brands()] == [first.id, second.id]


def test_update_brand():
    brand = _make_brand()
    updated = update_brand(brand.id, name="Acme Co", background="Updated", voice="", audience="")
    assert updated.name == "Acme Co"
    assert updated.background == "Updated"


def test_delete_brand_cascades_settings_and_history():
    brand = _make_brand()
    add_content_history(brand.id, "bluesky", BlueskyContent(posts=[BlueskyPost(text="hi")]))

    delete_brand(brand.id)

    assert get_brand(brand.id) is None
    assert list_content_history(brand.id, "bluesky") == []


def test_new_brand_has_default_content_settings():
    brand = _make_brand()
    assert get_bluesky_settings(brand.id).instructions == ""
    assert get_bluesky_settings(brand.id).hashtags == ""
    assert get_newsletter_settings(brand.id).instructions == ""
    assert get_newsletter_settings(brand.id).html_template == ""
    assert get_linkedin_settings(brand.id).instructions == ""


def test_update_bluesky_settings():
    brand = _make_brand()
    update_bluesky_settings(brand.id, instructions="No emojis", hashtags="#widgets")
    fetched = get_bluesky_settings(brand.id)
    assert fetched.instructions == "No emojis"
    assert fetched.hashtags == "#widgets"


def test_update_newsletter_settings():
    brand = _make_brand()
    update_newsletter_settings(
        brand.id,
        instructions="3 sections, 200 words each",
        html_template="<html><body>{{content}}</body></html>",
    )
    fetched = get_newsletter_settings(brand.id)
    assert fetched.instructions == "3 sections, 200 words each"
    assert fetched.html_template == "<html><body>{{content}}</body></html>"


def test_update_linkedin_settings():
    brand = _make_brand()
    update_linkedin_settings(brand.id, instructions="Always end with a question")
    assert get_linkedin_settings(brand.id).instructions == "Always end with a question"


def test_update_linkedin_settings_upserts_for_brand_missing_a_row():
    """linkedin_settings is a table added after brands could already exist —
    simulate a pre-existing brand with no row there yet (unlike
    create_brand(), which always inserts one) and confirm update still
    persists instead of silently affecting zero rows.
    """
    brand = _make_brand()
    conn = get_connection()
    conn.execute("DELETE FROM linkedin_settings WHERE brand_id = ?", (brand.id,))
    conn.commit()
    conn.close()

    update_linkedin_settings(brand.id, instructions="Backfilled brand")
    assert get_linkedin_settings(brand.id).instructions == "Backfilled brand"


def test_add_content_history_and_list_most_recent_first():
    brand = _make_brand()
    entry1 = add_content_history(
        brand.id, "bluesky", BlueskyContent(posts=[BlueskyPost(text="first")])
    )
    entry2 = add_content_history(
        brand.id, "bluesky", BlueskyContent(posts=[BlueskyPost(text="second")])
    )

    history = list_content_history(brand.id, "bluesky")
    assert [e.id for e in history] == [entry2.id, entry1.id]


def test_content_history_is_scoped_by_content_type():
    brand = _make_brand()
    add_content_history(brand.id, "bluesky", BlueskyContent(posts=[BlueskyPost(text="a post")]))
    add_content_history(
        brand.id,
        "newsletter",
        NewsletterContent(
            body_html="<p>hi</p>",
            subject_pairs=[SubjectPair(subject=f"s{i}", description=f"d{i}") for i in range(5)],
        ),
    )

    assert len(list_content_history(brand.id, "bluesky")) == 1
    assert len(list_content_history(brand.id, "newsletter")) == 1


def test_content_history_fifo_trim_at_20():
    brand = _make_brand()
    for i in range(25):
        add_content_history(
            brand.id, "bluesky", BlueskyContent(posts=[BlueskyPost(text=f"post {i}")])
        )

    history = list_content_history(brand.id, "bluesky")
    assert len(history) == 20
    # Most recent first, oldest 5 (post 0..4) trimmed away.
    assert history[0].payload.posts[0].text == "post 24"
    assert history[-1].payload.posts[0].text == "post 5"


def test_delete_single_content_history_entry():
    brand = _make_brand()
    entry = add_content_history(brand.id, "bluesky", BlueskyContent(posts=[BlueskyPost(text="hi")]))
    delete_content_history_entry(brand.id, entry.id)
    assert list_content_history(brand.id, "bluesky") == []


def test_delete_content_history_entry_scoped_to_brand():
    brand_a = _make_brand("A")
    brand_b = _make_brand("B")
    entry = add_content_history(
        brand_a.id, "bluesky", BlueskyContent(posts=[BlueskyPost(text="hi")])
    )

    # brand_b cannot delete brand_a's entry by guessing the id.
    delete_content_history_entry(brand_b.id, entry.id)

    assert len(list_content_history(brand_a.id, "bluesky")) == 1


def test_purge_content_history():
    brand = _make_brand()
    add_content_history(brand.id, "bluesky", BlueskyContent(posts=[BlueskyPost(text="one")]))
    add_content_history(brand.id, "bluesky", BlueskyContent(posts=[BlueskyPost(text="two")]))

    purge_content_history(brand.id, "bluesky")

    assert list_content_history(brand.id, "bluesky") == []


def test_recent_bluesky_texts_flattens_threads():
    brand = _make_brand()
    add_content_history(
        brand.id,
        "bluesky",
        BlueskyContent(posts=[BlueskyPost(text="part one"), BlueskyPost(text="part two")]),
    )
    texts = recent_bluesky_texts(brand.id)
    assert texts == ["part one / part two"]


def test_content_history_accepts_linkedin_end_to_end():
    brand = _make_brand()
    entry = add_content_history(brand.id, "linkedin", LinkedInContent(post_text="my post"))
    assert entry.content_type == "linkedin"

    history = list_content_history(brand.id, "linkedin")
    assert len(history) == 1
    assert history[0].payload.post_text == "my post"

    delete_content_history_entry(brand.id, entry.id)
    assert list_content_history(brand.id, "linkedin") == []


def test_recent_linkedin_texts_most_recent_first():
    brand = _make_brand()
    add_content_history(brand.id, "linkedin", LinkedInContent(post_text="first post"))
    add_content_history(brand.id, "linkedin", LinkedInContent(post_text="second post"))
    assert recent_linkedin_texts(brand.id) == ["second post", "first post"]


def test_recent_newsletter_summaries_strips_html():
    brand = _make_brand()
    add_content_history(
        brand.id,
        "newsletter",
        NewsletterContent(
            body_html="<p>Big <b>news</b> this week!</p>",
            subject_pairs=[SubjectPair(subject="Big News", description="d") for _ in range(5)],
        ),
    )
    summaries = recent_newsletter_summaries(brand.id)
    assert len(summaries) == 1
    assert "Big News" in summaries[0]
    assert "<p>" not in summaries[0]
    assert "Big news this week" in summaries[0]


def test_init_db_migrates_existing_db_missing_html_template_column(tmp_path, monkeypatch):
    """A real db predates the html_template column (added after initial
    launch) — init_db() must add it via ALTER TABLE without wiping existing
    brand data, since there's no separate migration tool for this app.

    Uses its own fresh, not-yet-initialized db file — the autouse temp_db
    fixture already ran init_db() against the usual per-test path, so this
    test points settings.db_path at a second file to simulate a genuinely
    pre-migration db (CREATE TABLE would otherwise collide with tables the
    fixture already created).
    """
    pre_migration_db = tmp_path / "pre_migration.db"
    monkeypatch.setattr(app_settings, "db_path", str(pre_migration_db))

    # Build the pre-migration schema by hand: same as schema.sql's
    # newsletter_settings, minus the html_template column.
    conn = sqlite3.connect(app_settings.db_path)
    conn.execute(
        """CREATE TABLE brands (
               id TEXT PRIMARY KEY, name TEXT NOT NULL, background TEXT NOT NULL DEFAULT '',
               voice TEXT NOT NULL DEFAULT '', audience TEXT NOT NULL DEFAULT '',
               skypilot_id TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE newsletter_settings (
               brand_id TEXT PRIMARY KEY REFERENCES brands (id) ON DELETE CASCADE,
               instructions TEXT NOT NULL DEFAULT ''
           )"""
    )
    conn.execute(
        "INSERT INTO brands (id, name, created_at) "
        "VALUES ('b1', 'Pre-existing Brand', '2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO newsletter_settings (brand_id, instructions) VALUES ('b1', 'old instructions')"
    )
    conn.commit()
    conn.close()

    init_db()

    brand = get_brand("b1")
    assert brand is not None
    assert brand.name == "Pre-existing Brand"
    fetched = get_newsletter_settings("b1")
    assert fetched.instructions == "old instructions"
    assert fetched.html_template == ""


def test_init_db_migrates_content_history_check_constraint_for_linkedin(tmp_path, monkeypatch):
    """SQLite bakes a CHECK constraint's value list into the table at
    creation time — a db predating LinkedIn support has content_history
    with CHECK (content_type IN ('bluesky', 'newsletter')), which would
    reject any 'linkedin' insert until the table is rebuilt. Confirmed by
    hand against a real db before this fix existed: the insert below
    failed with sqlite3.IntegrityError pre-migration. See
    db/connection.py's _recreate_content_history_with_check.
    """
    pre_migration_db = tmp_path / "pre_migration.db"
    monkeypatch.setattr(app_settings, "db_path", str(pre_migration_db))

    conn = sqlite3.connect(app_settings.db_path)
    conn.execute(
        """CREATE TABLE brands (
               id TEXT PRIMARY KEY, name TEXT NOT NULL, background TEXT NOT NULL DEFAULT '',
               voice TEXT NOT NULL DEFAULT '', audience TEXT NOT NULL DEFAULT '',
               skypilot_id TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE content_history (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               brand_id TEXT NOT NULL REFERENCES brands (id) ON DELETE CASCADE,
               content_type TEXT NOT NULL CHECK (content_type IN ('bluesky', 'newsletter')),
               payload TEXT NOT NULL, skypilot_post_id TEXT, scheduled_for TEXT, created_at TEXT NOT NULL
           )"""
    )
    conn.execute(
        "INSERT INTO brands (id, name, created_at) VALUES ('b1', 'Pre-existing Brand', '2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO content_history (brand_id, content_type, payload, created_at) "
        "VALUES ('b1', 'bluesky', '{\"posts\": [{\"text\": \"old post\"}]}', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    init_db()

    # Old row survived the table rebuild.
    history = list_content_history("b1", "bluesky")
    assert len(history) == 1

    # A linkedin insert, rejected by the old CHECK, now succeeds.
    add_content_history("b1", "linkedin", LinkedInContent(post_text="new post"))
    assert len(list_content_history("b1", "linkedin")) == 1

    # Idempotent — a second init_db() doesn't error or duplicate data.
    init_db()
    assert len(list_content_history("b1", "bluesky")) == 1
    assert len(list_content_history("b1", "linkedin")) == 1
