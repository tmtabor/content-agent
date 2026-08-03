"""LinkedIn post generation — a guided, multi-step wizard rather than a
single generate call: concept -> hooks -> hashtags -> outline -> draft ->
final polish. Each step is its own Agent, all sharing `LinkedInDeps`.

Output schemas follow the "keep it as flat as the data allows" lesson from
agent/agents/bluesky.py and agent/agents/newsletter.py: hooks/hashtags are
lists of small 2-field objects, the outline is a flat `list[str]`, and the
draft uses a bare `output_type=str` — no wrapper at all.

A second, less obvious lesson came out of evals/test_reliability.py while
building this agent: shape alone doesn't predict reliability — `HookOption`/
`HashtagOption` are the same 2-field-object-in-a-list shape as `SubjectPair`
in agent/agents/newsletter.py (which was reliable from the start), yet both
initially failed against the real model, repeatedly, with the model
inventing a *different* plausible-sounding key each time (`appeal`, then
`explanation`; `tag`, then `hashtag`) instead of using the schema's actual
field names. The fix wasn't renaming alone — it was rewriting each
instructions prompt to literally spell out the field names in prose (e.g.
"produce exactly 5 hook_text/explanation pairs"), the same way
`SubjectPair`'s prompt already said "subject-line/preview-description
pairs". If a future schema change here (or on a new agent) starts failing
with a `Field required` validation error naming an unexpected key, check
whether the instructions prose actually uses the schema's field names
before assuming the shape itself needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from agent.config import settings
from agent.logging import get_logger
from db.models import Brand, LinkedInSettings

logger = get_logger(__name__)

# Guardrail against runaway agentic loops — see CLAUDE.md.
USAGE_LIMITS = UsageLimits(request_limit=10, total_tokens_limit=100_000)

# The established gemma4 tuning (see CLAUDE.md) — retries=3 gives a small
# local model room to self-correct instead of hard-failing on one slip;
# temperature 0.6 (down from gemma4's default of 1.0) measurably reduces how
# often it drifts off the structured-output format. Shared by all 5 agents.
_MODEL_SETTINGS = {"temperature": 0.6}
_RETRIES = 3

# Against the real model, list-shaped outputs occasionally came back as
# conversational prose (a bulleted markdown list explaining itself) instead
# of the structured tool call at all — a different failure mode than the
# field-naming issue above, caught by the same eval. Appended to every
# agent's instructions as a blunt, explicit reminder.
_NO_PROSE_REMINDER = (
    "\nRespond only via the structured output — no explanatory prose, "
    "preamble, or commentary before or after it."
)

# Added after a real walkthrough (see logs/debug.log analysis) showed the
# outline step producing a single markdown-formatted blob — headers, **bold**,
# nested sub-bullets — which the draft step then inherited wholesale, forcing
# several rounds of manual regeneration to strip out. This brand's own recent
# posts (in context every call, via _shared_context_lines below) already
# demonstrate the correct Unicode-bold style with zero markdown, but the model
# didn't generalize from the examples alone — spelled out explicitly here on
# every agent whose output becomes visible post text.
_NO_MARKDOWN_REMINDER = (
    "\n\nLinkedIn renders plain text only — it does not support Markdown. Never use "
    "**bold**, *italic*, #headers, or -/* list syntax. For emphasis, use Unicode "
    "bold/italic character substitutes instead (e.g. 𝗕𝗼𝗹𝗱 𝗧𝗲𝘅𝘁), sparingly."
)


# --- Shared building blocks ---


class HookOption(BaseModel):
    """Field names went through two rounds of tuning against the real model
    (see evals/test_reliability.py): `text`/`appeal` caused it to put the
    actual hook copy into `appeal` and leave `text` missing; `why_it_works`
    fared no better — it kept substituting `explanation` instead. `subject`/
    `description` in `SubjectPair` (agent/agents/newsletter.py) never had
    this problem. The difference: that model's *instructions prose* literally
    says "subject-line/preview-description pairs", echoing the exact field
    names, so the model reuses them naturally instead of guessing a
    synonym — the instructions below for `hook_text`/`explanation` now do
    the same. Field-name ambiguity was the real cause, not the 2-field-
    object shape itself; matching the prose to the schema is what fixed it,
    not the specific words chosen.
    """

    hook_text: str = Field(
        description="The hook's exact wording, ready to use as the post's opening line(s)."
    )
    explanation: str = Field(
        description="A short explanation of why this hook is effective, e.g. 'curiosity gap'."
    )


class HashtagOption(BaseModel):
    """`hashtag`, not `tag` — evals/test_reliability.py caught the model
    reliably preferring the more natural key name `hashtag` over `tag`
    (same class of field-naming ambiguity as HookOption above, same fix).
    """

    hashtag: str = Field(description="The hashtag itself, including the # symbol.")
    description: str = Field(description="What audience or topic this hashtag targets.")


class PolishResult(BaseModel):
    corrected_text: str = Field(
        description="The full post text after proofreading, with any fixes applied. "
        "Identical to the original if nothing needed to change."
    )
    changes: list[str] = Field(
        description="Short, specific bullet points describing each change made. "
        "Empty list if no changes were made."
    )


class LinkedInWizardState(BaseModel):
    """The whole wizard's state — JSON-serialized into a single hidden form
    field by web/routers/linkedin.py, round-tripped through every step
    (see that module for the caching/back-navigation rules built on top of
    this). Lives here, not in web/, because it's built from the same
    output types the agents below produce.
    """

    step: Literal["concept", "hooks", "hashtags", "outline", "draft", "polish"] = "concept"
    concept: str = ""

    hook_options: list[HookOption] = Field(default_factory=list)
    accepted_hook: str = ""

    hashtag_options: list[HashtagOption] = Field(default_factory=list)
    accepted_hashtags: list[str] = Field(default_factory=list)

    outline_bullets: list[str] = Field(default_factory=list)

    draft_text: str = ""

    polish_original: str = ""  # draft_text frozen at the moment polish ran
    polish_result: PolishResult | None = None

    # Every regenerate's instructions, cumulative across all steps — the
    # spec's "the agent retains in its context every accepted step and
    # feedback" applies to feedback too, not just accepted content, so this
    # is never cleared and is included in every subsequent agent call.
    feedback_log: list[str] = Field(default_factory=list)


@dataclass
class LinkedInDeps:
    brand: Brand
    settings: LinkedInSettings
    recent_posts: list[str] = field(default_factory=list)  # cross-session anti-repetition
    concept: str = ""
    accepted_hook: str = ""
    accepted_hashtags: list[str] = field(default_factory=list)
    outline_bullets: list[str] = field(default_factory=list)
    draft_text: str = ""
    feedback_log: list[str] = field(default_factory=list)
    # Regeneration-only: the options being replaced (dedupe context) and any
    # fresh instructions for this specific call. The caller is responsible
    # for appending regen_instructions to feedback_log before the call, so
    # it also persists for later steps — see web/routers/linkedin.py.
    previous_options: list[str] = field(default_factory=list)
    regen_instructions: str = ""


def _shared_context_lines(deps: LinkedInDeps) -> list[str]:
    """Brand identity + LinkedIn instructions + anti-repetition + the full
    cumulative feedback log — common to all 5 agents below.
    """
    brand = deps.brand
    lines = [
        f"Brand: {brand.name}",
        f"Background: {brand.background}" if brand.background else "",
        f"Voice/tone: {brand.voice}" if brand.voice else "",
        f"Audience: {brand.audience}" if brand.audience else "",
    ]
    if deps.settings.instructions:
        lines.append(f"LinkedIn-specific instructions: {deps.settings.instructions}")
    if deps.recent_posts:
        recent = "\n".join(f"- {post}" for post in deps.recent_posts)
        lines.append(f"Recent LinkedIn posts for this brand (do not repeat these):\n{recent}")
    if deps.feedback_log:
        feedback = "\n".join(f"- {note}" for note in deps.feedback_log)
        lines.append(
            f"Feedback/instructions given earlier in this session (all still apply):\n{feedback}"
        )
    return lines


# --- 1. Hooks ---

HooksOutput = Annotated[list[HookOption], Field(min_length=5, max_length=5)]

hooks_agent: Agent[LinkedInDeps, HooksOutput] = Agent(
    settings.model,
    name="linkedin_hooks_agent",
    output_type=HooksOutput,
    deps_type=LinkedInDeps,
    retries=_RETRIES,
    model_settings=_MODEL_SETTINGS,
    instructions="""You write opening hooks for a LinkedIn post.

Given a content idea, produce exactly 5 distinct hook_text/explanation pairs. hook_text
is the first line(s) that stop someone scrolling — high-performing, ready to use as
written. explanation is a short note on why that hook works (e.g. "curiosity gap",
"contrarian claim", "specific number"). The 5 hooks should take genuinely different
angles, not reword the same idea."""
    + _NO_MARKDOWN_REMINDER
    + _NO_PROSE_REMINDER,
)


@hooks_agent.instructions
async def hooks_context(ctx: RunContext[LinkedInDeps]) -> str:
    deps = ctx.deps
    lines = [*_shared_context_lines(deps), f"Content idea: {deps.concept}"]
    if deps.previous_options:
        previous = "\n".join(f"- {text}" for text in deps.previous_options)
        lines.append(f"Already suggested — do not repeat these hooks:\n{previous}")
    if deps.regen_instructions:
        lines.append(f"Additional instructions for this regeneration: {deps.regen_instructions}")
    return "\n".join(line for line in lines if line)


async def run_linkedin_hooks_agent(deps: LinkedInDeps) -> list[HookOption]:
    logger.info("Running LinkedIn hooks agent", extra={"brand_id": deps.brand.id})
    result = await hooks_agent.run(deps.concept, deps=deps, usage_limits=USAGE_LIMITS)
    # Only visible with AGENT_DEBUG=true (see agent/logging.py) — nothing
    # else captures the actual generated content on a *successful* call.
    logger.debug("LinkedIn hooks agent output", extra={"hooks": result.output})
    return result.output


# --- 2. Hashtags ---

HashtagsOutput = Annotated[list[HashtagOption], Field(min_length=10, max_length=10)]

hashtags_agent: Agent[LinkedInDeps, HashtagsOutput] = Agent(
    settings.model,
    name="linkedin_hashtags_agent",
    output_type=HashtagsOutput,
    deps_type=LinkedInDeps,
    retries=_RETRIES,
    model_settings=_MODEL_SETTINGS,
    instructions="""You suggest hashtags for a LinkedIn post.

Given the post's content idea and chosen hook, produce exactly 10 hashtag/description
pairs. hashtag is a relevant, high-traffic LinkedIn hashtag (including the # symbol).
description is a short note on what audience or topic it targets. Prefer a mix of broad
and niche hashtags over 10 near-duplicates."""
    + _NO_PROSE_REMINDER,
)


@hashtags_agent.instructions
async def hashtags_context(ctx: RunContext[LinkedInDeps]) -> str:
    deps = ctx.deps
    lines = [
        *_shared_context_lines(deps),
        f"Content idea: {deps.concept}",
        f"Chosen hook: {deps.accepted_hook}",
    ]
    if deps.previous_options:
        previous = "\n".join(f"- {tag}" for tag in deps.previous_options)
        lines.append(f"Already suggested — do not repeat these hashtags:\n{previous}")
    if deps.regen_instructions:
        lines.append(f"Additional instructions for this regeneration: {deps.regen_instructions}")
    return "\n".join(line for line in lines if line)


async def run_linkedin_hashtags_agent(deps: LinkedInDeps) -> list[HashtagOption]:
    logger.info("Running LinkedIn hashtags agent", extra={"brand_id": deps.brand.id})
    result = await hashtags_agent.run(deps.concept, deps=deps, usage_limits=USAGE_LIMITS)
    logger.debug("LinkedIn hashtags agent output", extra={"hashtags": result.output})
    return result.output


# --- 3. Outline ---

# Unconstrained list[str] previously let the model satisfy the schema with a
# single list item containing the entire post (headers, nested sub-bullets,
# a closing CTA, hashtags) — technically valid, but not an outline, and the
# root cause of the markdown-formatting problem described at
# _NO_MARKDOWN_REMINDER above. min/max pushes toward what "a bulleted
# outline" actually means: several short, distinct points, not one blob.
OutlineOutput = Annotated[list[str], Field(min_length=3, max_length=8)]

outline_agent: Agent[LinkedInDeps, OutlineOutput] = Agent(
    settings.model,
    name="linkedin_outline_agent",
    output_type=OutlineOutput,
    deps_type=LinkedInDeps,
    retries=_RETRIES,
    model_settings=_MODEL_SETTINGS,
    instructions="""You outline a LinkedIn post.

Given the content idea, chosen hook, and chosen hashtags, produce between 3 and 8
short bullet points — the structure and key points the full post will cover, not the
post itself. Each bullet is its own short list item: a phrase or one sentence, never
a paragraph, and never multiple points combined into a single item. Start from the
hook and build toward a clear takeaway or call to action.

If a current outline is given in the context below, you are revising it, not writing
a new one from scratch — apply the feedback given and leave everything the feedback
doesn't address unchanged."""
    + _NO_MARKDOWN_REMINDER
    + _NO_PROSE_REMINDER,
)


@outline_agent.instructions
async def outline_context(ctx: RunContext[LinkedInDeps]) -> str:
    deps = ctx.deps
    lines = [
        *_shared_context_lines(deps),
        f"Content idea: {deps.concept}",
        f"Chosen hook: {deps.accepted_hook}",
        f"Chosen hashtags: {', '.join(deps.accepted_hashtags)}",
    ]
    if deps.outline_bullets:
        # Present during regeneration only (empty on first generation) — the
        # anchor that turns "regenerate with feedback" into an actual
        # revision instead of a blind rewrite from scratch. See
        # web/routers/linkedin.py's regenerate_outline().
        current = "\n".join(f"- {bullet}" for bullet in deps.outline_bullets)
        lines.append(
            "Current outline — revise it based on the feedback given, keeping "
            f"anything the feedback doesn't address unchanged:\n{current}"
        )
    return "\n".join(line for line in lines if line)


async def run_linkedin_outline_agent(deps: LinkedInDeps) -> list[str]:
    logger.info("Running LinkedIn outline agent", extra={"brand_id": deps.brand.id})
    result = await outline_agent.run(deps.concept, deps=deps, usage_limits=USAGE_LIMITS)
    logger.debug("LinkedIn outline agent output", extra={"outline": result.output})
    return result.output


# --- 4. Draft ---

draft_agent: Agent[LinkedInDeps, str] = Agent(
    settings.model,
    name="linkedin_draft_agent",
    output_type=str,
    deps_type=LinkedInDeps,
    retries=_RETRIES,
    model_settings=_MODEL_SETTINGS,
    instructions="""You draft a full LinkedIn post.

Given the content idea, chosen hook, chosen hashtags, and outline, write the full post
text — starting with the hook, following the outline's structure, ending with the
hashtags. Write in the brand's voice, for its audience. Return only the post text.

The outline is the complete content plan — cover only the points it contains. Brand
background is for voice, tone, and factual grounding on those points, not a source of
additional topics or facts beyond what the outline covers, even if they're mentioned
in the background above.

If a current draft is given in the context below, you are revising it, not writing a
new one from scratch — apply the feedback given and leave everything the feedback
doesn't address unchanged."""
    + _NO_MARKDOWN_REMINDER
    + _NO_PROSE_REMINDER,
)


@draft_agent.instructions
async def draft_context(ctx: RunContext[LinkedInDeps]) -> str:
    deps = ctx.deps
    bullets = "\n".join(f"- {bullet}" for bullet in deps.outline_bullets)
    lines = [
        *_shared_context_lines(deps),
        f"Content idea: {deps.concept}",
        f"Chosen hook: {deps.accepted_hook}",
        f"Chosen hashtags: {', '.join(deps.accepted_hashtags)}",
        f"Outline:\n{bullets}",
    ]
    if deps.draft_text:
        # Present during regeneration only (empty on first generation) —
        # same reasoning as outline_context above. See
        # web/routers/linkedin.py's regenerate_draft().
        lines.append(
            "Current draft — revise it based on the feedback given, keeping anything "
            f"the feedback doesn't address unchanged:\n{deps.draft_text}"
        )
    return "\n".join(line for line in lines if line)


async def run_linkedin_draft_agent(deps: LinkedInDeps) -> str:
    logger.info("Running LinkedIn draft agent", extra={"brand_id": deps.brand.id})
    result = await draft_agent.run(deps.concept, deps=deps, usage_limits=USAGE_LIMITS)
    logger.debug("LinkedIn draft agent output", extra={"draft": result.output})
    return result.output


# --- 5. Polish (final pass) ---

polish_agent: Agent[LinkedInDeps, PolishResult] = Agent(
    settings.model,
    name="linkedin_polish_agent",
    output_type=PolishResult,
    deps_type=LinkedInDeps,
    retries=_RETRIES,
    model_settings=_MODEL_SETTINGS,
    instructions="""You proofread and tighten a LinkedIn post.

Given the drafted post, look for typos, grammar issues, and places where the post is
weak, and produce a corrected version (corrected_text). If the drafted post contains
Markdown syntax (**bold**, *italic*, #headers, -/* lists), replace it with Unicode
bold/italic character substitutes or plain text — this counts as a fix, note it in
changes. If nothing needs to change, corrected_text can be identical to the original.
List every change you made as short, specific bullet points (changes) — e.g. "Fixed
'recieve' -> 'receive'", "Shortened the third paragraph for punchier flow", "Replaced
markdown bold with Unicode bold". An empty changes list means no changes were made."""
    + _NO_MARKDOWN_REMINDER
    + _NO_PROSE_REMINDER,
)


@polish_agent.instructions
async def polish_context(ctx: RunContext[LinkedInDeps]) -> str:
    deps = ctx.deps
    lines = [*_shared_context_lines(deps), f"Drafted post to review:\n{deps.draft_text}"]
    return "\n".join(line for line in lines if line)


async def run_linkedin_polish_agent(deps: LinkedInDeps) -> PolishResult:
    logger.info("Running LinkedIn polish agent", extra={"brand_id": deps.brand.id})
    result = await polish_agent.run(deps.draft_text, deps=deps, usage_limits=USAGE_LIMITS)
    logger.debug("LinkedIn polish agent output", extra={"polish_result": result.output})
    return result.output
