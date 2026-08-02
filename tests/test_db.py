"""Unit tests for the SQLite repository layer — brand CRUD, per-content-type
settings, and content history (FIFO trim, purge).
"""

import sqlite3

from agent.config import settings as app_settings
from db.connection import init_db
from db.models import BlueskyContent, BlueskyPost, NewsletterContent, SubjectPair
from db.repository import (
    add_content_history,
    create_brand,
    delete_brand,
    delete_content_history_entry,
    get_bluesky_settings,
    get_brand,
    get_newsletter_settings,
    list_brands,
    list_content_history,
    purge_content_history,
    recent_bluesky_texts,
    recent_newsletter_summaries,
    update_bluesky_settings,
    update_brand,
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
