"""CRUD access to the SQLite file — brands, per-content-type settings, and history.

Every function opens and closes its own connection via `db_session()`. Data
volume here is trivial (one brand's config, at most 20 history rows per
brand+content-type) so there's no pooling or async driver — see the plan's
rationale for keeping this synchronous.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import UTC, datetime

from db.connection import db_session
from db.models import (
    BlueskyContent,
    BlueskySettings,
    Brand,
    ContentHistoryEntry,
    ContentType,
    NewsletterContent,
    NewsletterSettings,
)

# Only the most recent N pieces of content are kept per brand+content-type,
# FIFO — older entries are deleted on insert so the "recent" context fed
# back into generation never grows unbounded.
HISTORY_LIMIT = 20

_PAYLOAD_MODELS: dict[ContentType, type[BlueskyContent] | type[NewsletterContent]] = {
    "bluesky": BlueskyContent,
    "newsletter": NewsletterContent,
}


def _row_to_brand(row: sqlite3.Row) -> Brand:
    return Brand(
        id=row["id"],
        name=row["name"],
        background=row["background"],
        voice=row["voice"],
        audience=row["audience"],
        skypilot_id=row["skypilot_id"],
        created_at=row["created_at"],
    )


def _row_to_history_entry(row: sqlite3.Row) -> ContentHistoryEntry:
    content_type: ContentType = row["content_type"]
    payload_model = _PAYLOAD_MODELS[content_type]
    return ContentHistoryEntry(
        id=row["id"],
        brand_id=row["brand_id"],
        content_type=content_type,
        payload=payload_model.model_validate_json(row["payload"]),
        skypilot_post_id=row["skypilot_post_id"],
        scheduled_for=row["scheduled_for"],
        created_at=row["created_at"],
    )


# --- Brands ---


def create_brand(
    name: str,
    background: str = "",
    voice: str = "",
    audience: str = "",
    skypilot_id: str = "",
) -> Brand:
    brand_id = uuid.uuid4().hex
    created_at = datetime.now(UTC).isoformat()
    with db_session() as conn:
        conn.execute(
            """INSERT INTO brands (id, name, background, voice, audience, skypilot_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (brand_id, name, background, voice, audience, skypilot_id, created_at),
        )
        conn.execute("INSERT INTO bluesky_settings (brand_id) VALUES (?)", (brand_id,))
        conn.execute("INSERT INTO newsletter_settings (brand_id) VALUES (?)", (brand_id,))
    return Brand(
        id=brand_id,
        name=name,
        background=background,
        voice=voice,
        audience=audience,
        skypilot_id=skypilot_id,
        created_at=created_at,
    )


def get_brand(brand_id: str) -> Brand | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM brands WHERE id = ?", (brand_id,)).fetchone()
    return _row_to_brand(row) if row else None


def list_brands() -> list[Brand]:
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM brands ORDER BY created_at ASC").fetchall()
    return [_row_to_brand(row) for row in rows]


def update_brand(
    brand_id: str,
    name: str,
    background: str = "",
    voice: str = "",
    audience: str = "",
    skypilot_id: str = "",
) -> Brand | None:
    with db_session() as conn:
        conn.execute(
            """UPDATE brands SET name = ?, background = ?, voice = ?, audience = ?, skypilot_id = ?
               WHERE id = ?""",
            (name, background, voice, audience, skypilot_id, brand_id),
        )
    return get_brand(brand_id)


def delete_brand(brand_id: str) -> None:
    """Delete a brand. Settings and history rows cascade via foreign keys."""
    with db_session() as conn:
        conn.execute("DELETE FROM brands WHERE id = ?", (brand_id,))


# --- Per-content-type settings ---


def get_bluesky_settings(brand_id: str) -> BlueskySettings:
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM bluesky_settings WHERE brand_id = ?", (brand_id,)
        ).fetchone()
    if row is None:
        return BlueskySettings(brand_id=brand_id)
    return BlueskySettings(
        brand_id=row["brand_id"], instructions=row["instructions"], hashtags=row["hashtags"]
    )


def update_bluesky_settings(brand_id: str, instructions: str, hashtags: str) -> BlueskySettings:
    with db_session() as conn:
        conn.execute(
            "UPDATE bluesky_settings SET instructions = ?, hashtags = ? WHERE brand_id = ?",
            (instructions, hashtags, brand_id),
        )
    return BlueskySettings(brand_id=brand_id, instructions=instructions, hashtags=hashtags)


def get_newsletter_settings(brand_id: str) -> NewsletterSettings:
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM newsletter_settings WHERE brand_id = ?", (brand_id,)
        ).fetchone()
    if row is None:
        return NewsletterSettings(brand_id=brand_id)
    return NewsletterSettings(
        brand_id=row["brand_id"],
        instructions=row["instructions"],
        html_template=row["html_template"],
    )


def update_newsletter_settings(
    brand_id: str, instructions: str, html_template: str = ""
) -> NewsletterSettings:
    with db_session() as conn:
        conn.execute(
            "UPDATE newsletter_settings SET instructions = ?, html_template = ? WHERE brand_id = ?",
            (instructions, html_template, brand_id),
        )
    return NewsletterSettings(
        brand_id=brand_id, instructions=instructions, html_template=html_template
    )


# --- Content history ---


def add_content_history(
    brand_id: str,
    content_type: ContentType,
    payload: BlueskyContent | NewsletterContent,
    skypilot_post_id: str | None = None,
    scheduled_for: datetime | None = None,
) -> ContentHistoryEntry:
    """Insert a new history entry, then trim to the most recent HISTORY_LIMIT (FIFO)."""
    created_at = datetime.now(UTC).isoformat()
    with db_session() as conn:
        cursor = conn.execute(
            """INSERT INTO content_history
                   (brand_id, content_type, payload, skypilot_post_id, scheduled_for, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                brand_id,
                content_type,
                payload.model_dump_json(),
                skypilot_post_id,
                scheduled_for.isoformat() if scheduled_for else None,
                created_at,
            ),
        )
        new_id = cursor.lastrowid
        conn.execute(
            """DELETE FROM content_history
               WHERE brand_id = ? AND content_type = ? AND id NOT IN (
                   SELECT id FROM content_history
                   WHERE brand_id = ? AND content_type = ?
                   ORDER BY created_at DESC, id DESC
                   LIMIT ?
               )""",
            (brand_id, content_type, brand_id, content_type, HISTORY_LIMIT),
        )
        row = conn.execute("SELECT * FROM content_history WHERE id = ?", (new_id,)).fetchone()
    return _row_to_history_entry(row)


def list_content_history(brand_id: str, content_type: ContentType) -> list[ContentHistoryEntry]:
    """Most-recent-first."""
    with db_session() as conn:
        rows = conn.execute(
            """SELECT * FROM content_history WHERE brand_id = ? AND content_type = ?
               ORDER BY created_at DESC, id DESC""",
            (brand_id, content_type),
        ).fetchall()
    return [_row_to_history_entry(row) for row in rows]


def delete_content_history_entry(brand_id: str, entry_id: int) -> None:
    """Scoped to brand_id so a request for one brand can't delete another's history."""
    with db_session() as conn:
        conn.execute(
            "DELETE FROM content_history WHERE id = ? AND brand_id = ?", (entry_id, brand_id)
        )


def purge_content_history(brand_id: str, content_type: ContentType) -> None:
    with db_session() as conn:
        conn.execute(
            "DELETE FROM content_history WHERE brand_id = ? AND content_type = ?",
            (brand_id, content_type),
        )


# --- Context helpers for generation prompts ---


def recent_bluesky_texts(brand_id: str, limit: int = HISTORY_LIMIT) -> list[str]:
    """Flatten each past thread/post into one string, most recent first."""
    entries = list_content_history(brand_id, "bluesky")[:limit]
    texts = []
    for entry in entries:
        assert isinstance(entry.payload, BlueskyContent)
        texts.append(" / ".join(post.text for post in entry.payload.posts))
    return texts


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_EXCERPT_LENGTH = 300


def _plain_text_excerpt(html: str, length: int = _EXCERPT_LENGTH) -> str:
    """Strip tags and collapse whitespace for use as LLM context, not display."""
    text = _WHITESPACE_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()
    return text[:length]


def recent_newsletter_summaries(brand_id: str, limit: int = HISTORY_LIMIT) -> list[str]:
    """Subject + a short plain-text excerpt per past newsletter, most recent first.

    Full HTML bodies aren't used here — up to 20 of them could easily blow
    past a local model's context window, so only a condensed summary goes
    into the "avoid repeating this" prompt context. The full body is still
    stored in content_history for the settings-page history browser.
    """
    entries = list_content_history(brand_id, "newsletter")[:limit]
    summaries = []
    for entry in entries:
        assert isinstance(entry.payload, NewsletterContent)
        subject = entry.payload.subject_pairs[0].subject if entry.payload.subject_pairs else ""
        excerpt = _plain_text_excerpt(entry.payload.body_html)
        summaries.append(f"{subject}: {excerpt}")
    return summaries
