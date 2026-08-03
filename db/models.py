"""Pydantic models for everything stored in the SQLite file.

`content_history.payload` is stored as JSON text because Bluesky content
(a list of short post texts) and newsletter content (an HTML body plus five
subject/description pairs) have genuinely different shapes — a generic JSON
column here means a future content type needs no schema migration, just a
new payload model and a literal added to `ContentType`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ContentType = Literal["bluesky", "newsletter", "linkedin"]


class Brand(BaseModel):
    id: str
    name: str
    background: str = ""
    voice: str = ""
    audience: str = ""
    skypilot_id: str = ""
    created_at: datetime


class BlueskySettings(BaseModel):
    brand_id: str
    instructions: str = ""
    hashtags: str = ""


class NewsletterSettings(BaseModel):
    brand_id: str
    instructions: str = ""
    html_template: str = ""


class BlueskyPost(BaseModel):
    """One post. A thread is a `BlueskyContent` with more than one of these."""

    text: str = Field(max_length=300)


class BlueskyContent(BaseModel):
    posts: list[BlueskyPost] = Field(min_length=1)


class SubjectPair(BaseModel):
    subject: str
    description: str


class NewsletterContent(BaseModel):
    body_html: str
    subject_pairs: list[SubjectPair] = Field(min_length=5, max_length=5)


class LinkedInSettings(BaseModel):
    brand_id: str
    instructions: str = ""


class LinkedInContent(BaseModel):
    """What "Mark as Used" saves — final text only, matching Bluesky/Newsletter
    precedent. The wizard's intermediate state (hooks, hashtags, outline,
    feedback log) lives only in the in-progress form, not in history.
    """

    post_text: str


class ContentHistoryEntry(BaseModel):
    id: int
    brand_id: str
    content_type: ContentType
    payload: BlueskyContent | NewsletterContent | LinkedInContent
    skypilot_post_id: str | None = None
    scheduled_for: datetime | None = None
    created_at: datetime
