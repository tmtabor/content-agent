"""Re-exports for the app's content agents.

The template's single-canonical-name convention (`run_agent`, `AgentOutput`,
`AgentDeps`, `agent` — see the deleted agent/agents/single.py) assumed one
agent per app. This app has three independent content agents now, each
exported under its own name instead of forcing a shared canonical name that
no longer fits. Newsletter generation is split into two smaller sequential
calls (body, then subjects) — see agent/agents/newsletter.py for why — and
LinkedIn generation into five (see agent/agents/linkedin.py), so all of
their Agent objects are exported here, not just one per module.
"""

from agent.agents.bluesky import (
    USAGE_LIMITS as BLUESKY_USAGE_LIMITS,
    BlueskyDeps,
    BlueskyOutput,
    bluesky_agent,
    run_bluesky_agent,
)
from agent.agents.linkedin import (
    USAGE_LIMITS as LINKEDIN_USAGE_LIMITS,
    HashtagOption,
    HookOption,
    LinkedInDeps,
    LinkedInWizardState,
    PolishResult,
    draft_agent as linkedin_draft_agent,
    hashtags_agent as linkedin_hashtags_agent,
    hooks_agent as linkedin_hooks_agent,
    outline_agent as linkedin_outline_agent,
    polish_agent as linkedin_polish_agent,
    run_linkedin_draft_agent,
    run_linkedin_hashtags_agent,
    run_linkedin_hooks_agent,
    run_linkedin_outline_agent,
    run_linkedin_polish_agent,
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
    "LINKEDIN_USAGE_LIMITS",
    "HashtagOption",
    "HookOption",
    "LinkedInDeps",
    "LinkedInWizardState",
    "PolishResult",
    "linkedin_draft_agent",
    "linkedin_hashtags_agent",
    "linkedin_hooks_agent",
    "linkedin_outline_agent",
    "linkedin_polish_agent",
    "run_linkedin_draft_agent",
    "run_linkedin_hashtags_agent",
    "run_linkedin_hooks_agent",
    "run_linkedin_outline_agent",
    "run_linkedin_polish_agent",
]
