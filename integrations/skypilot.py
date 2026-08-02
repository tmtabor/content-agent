"""SkyPilot API client — https://skypilot.social/static/html/api.html

Posting is a deterministic action the UI takes after a human approves a
generated draft, not something the LLM decides to do, so this is a plain
httpx client called directly from web/routers/bluesky.py rather than an
agent tool.

SKYPILOT_API_KEY is read lazily (not at import time, unlike the model
provider keys in agent/config.py) since it's only needed when the user
actually clicks "Send to SkyPilot" — an app that never posts shouldn't be
forced to have it configured.
"""

from __future__ import annotations

import os
from contextlib import suppress
from datetime import datetime

import httpx

API_BASE = "https://skypilot.social/api/v1"


class SkyPilotError(Exception):
    """Raised when the SkyPilot API rejects a request or is unreachable."""


def _api_key() -> str:
    key = os.getenv("SKYPILOT_API_KEY")
    if not key:
        raise SkyPilotError(
            "SKYPILOT_API_KEY is not set. Add it to .env to send posts to SkyPilot."
        )
    return key


async def create_post(
    account_id: str,
    texts: list[str],
    scheduled_for: datetime | None = None,
) -> dict:
    """Create a post (len(texts) == 1) or thread (len(texts) > 1).

    Args:
        account_id: The brand's SkyPilot account UUID.
        texts: Post text(s), in thread order. Each must already be
            <=300 characters — the caller (the Bluesky agent's output
            schema) enforces that, not this client.
        scheduled_for: When set, the post is scheduled for this time.
            When None, it's created as an unscheduled draft.

    Returns:
        The created PostThread as returned by the API (includes "id").

    Raises:
        SkyPilotError: On a missing API key, a non-2xx response, or a
            network failure.
    """
    payload: dict = {
        "account_id": account_id,
        "posts": [{"text": text, "order": i} for i, text in enumerate(texts)],
    }
    if scheduled_for is not None:
        payload["scheduled_for"] = scheduled_for.isoformat()

    headers = {"Authorization": f"Bearer {_api_key()}"}

    try:
        async with httpx.AsyncClient(base_url=API_BASE, timeout=30.0) as client:
            response = await client.post("/posts/", json=payload, headers=headers)
    except httpx.HTTPError as e:
        raise SkyPilotError(f"Could not reach SkyPilot: {e}") from e

    if response.status_code not in (200, 201):
        detail = response.text
        with suppress(ValueError):
            detail = response.json()
        raise SkyPilotError(f"SkyPilot API error ({response.status_code}): {detail}")

    return response.json()
