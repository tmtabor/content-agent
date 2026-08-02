"""Shared template context — the brand tab bar and content-agent nav need the
same brand list + active-state info on nearly every page.
"""

from db.repository import list_brands

# One entry per content agent, in left-nav order. Adding a future agent
# means adding an entry here plus a router and templates — nothing else in
# the base shell needs to change.
CONTENT_AGENTS = [
    {"key": "bluesky", "label": "Bluesky"},
    {"key": "newsletter", "label": "Newsletter"},
]


def base_context(active_brand_id: str | None, active_agent: str | None) -> dict:
    return {
        "brands": list_brands(),
        "active_brand_id": active_brand_id,
        "active_agent": active_agent,
        "content_agents": CONTENT_AGENTS,
    }
