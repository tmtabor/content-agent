"""Re-exports for the app's content agents.

The template's single-canonical-name convention (`run_agent`, `AgentOutput`,
`AgentDeps`, `agent` — see the deleted agent/agents/single.py) assumed one
agent per app. This app has two independent content agents now, with more
planned, so each is exported under its own name instead of forcing a shared
canonical name that no longer fits. Newsletter generation is itself split
into two smaller sequential calls (body, then subjects) — see
agent/agents/newsletter.py for why — so both of its Agent objects are
exported here, not just one.
"""

from agent.agents.bluesky import (
    USAGE_LIMITS as BLUESKY_USAGE_LIMITS,
    BlueskyDeps,
    BlueskyOutput,
    bluesky_agent,
    run_bluesky_agent,
)
from agent.agents.newsletter import (
    USAGE_LIMITS as NEWSLETTER_USAGE_LIMITS,
    NewsletterDeps,
    NewsletterOutput,
    SubjectPair,
    newsletter_body_agent,
    newsletter_subjects_agent,
    run_newsletter_agent,
)

__all__ = [
    "BLUESKY_USAGE_LIMITS",
    "BlueskyDeps",
    "BlueskyOutput",
    "bluesky_agent",
    "run_bluesky_agent",
    "NEWSLETTER_USAGE_LIMITS",
    "NewsletterDeps",
    "NewsletterOutput",
    "SubjectPair",
    "newsletter_body_agent",
    "newsletter_subjects_agent",
    "run_newsletter_agent",
]
