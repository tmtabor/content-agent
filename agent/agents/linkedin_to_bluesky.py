"""LinkedIn post -> one or more Bluesky posts/threads, split into two
sequential kinds of agent calls: first plan the breakdown (how many groups,
how big each is, what each covers), then write the actual post text — one
call per group. Same reliability rationale as agent/agents/newsletter.py's
body/subjects split and agent/agents/linkedin.py's outline/draft split —
see CLAUDE.md.

The write call is scoped to exactly one group per call, not the whole plan
at once. An earlier version made a single write call for the entire plan,
returning one flat `list[str]` re-grouped after the fact using the plan's
`group_sizes` — but that asks gemma4 to simultaneously hit several exact
counts within one continuous list (e.g. stop at 2, then stop at 3, then
stop at 1), and a real run logged in logs/debug.log showed it reliably
overshooting and exhausting all 3 retries, hard-failing the whole
generation. Splitting the write call per group reduces that to one exact
count per call — the same "smaller task, higher reliability" principle
behind the plan/write split itself, just applied one level deeper. Each
call still returns a flat `list[str]` (the same proven shape as
`BlueskyOutput.posts`), just for one group instead of all of them, so this
still never asks gemma4 for a nested list or list-of-objects, the class of
shape this project has repeatedly found unreliable (see
`HookOption`/`HashtagOption`'s docstrings in agent/agents/linkedin.py).
Groups are written sequentially, each one told what earlier groups in the
same plan already wrote, so independently-generated groups don't repeat the
same phrasing or framing.

Reuses `bluesky_settings` (brand voice/hashtags/instructions) and
`content_type="bluesky"` for history — no new settings table, no schema
changes. A "group" (`LinkedInToBlueskyGroup`) is an independent post
(`len(posts) == 1`) or a thread (`len(posts) > 1`); there's no separate
flag, it's purely derived from post count.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.usage import UsageLimits

from agent.config import settings
from agent.logging import get_logger
from db.models import BlueskySettings, Brand

logger = get_logger(__name__)

# Guardrail against runaway agentic loops — see CLAUDE.md. Each of the two
# calls below gets its own independent budget, same reasoning as
# agent/agents/newsletter.py's USAGE_LIMITS comment.
USAGE_LIMITS = UsageLimits(request_limit=10, total_tokens_limit=100_000)

# The established gemma4 tuning (see CLAUDE.md) — shared by both calls below.
_MODEL_SETTINGS = {"temperature": 0.6}
_RETRIES = 3


# --- Wizard state ---


class LinkedInToBlueskyGroup(BaseModel):
    """One independent post (`len(posts) == 1`) or one thread
    (`len(posts) > 1`) — no separate flag; a group's shape is purely its
    post count.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    posts: list[Annotated[str, Field(max_length=300)]] = Field(min_length=1)
    status: Literal["pending", "sent", "used"] = "pending"
    skypilot_post_id: str | None = None
    scheduled_for: datetime | None = None


class LinkedInToBlueskyWizardState(BaseModel):
    """JSON-serialized into one hidden form field, round-tripped through
    every request — same pattern as agent/agents/linkedin.py's
    `LinkedInWizardState`, needed here for the same reason: a variable
    number of independently editable/deletable/sendable groups doesn't fit
    a handful of flat hidden fields.
    """

    step: Literal["input", "results"] = "input"
    source_text: str = ""
    target_count: int | None = None
    thread_preference: Literal["no_preference", "independent", "thread"] = "no_preference"
    groups: list[LinkedInToBlueskyGroup] = Field(default_factory=list)
    # Cumulative across regenerates, like LinkedIn's feedback_log.
    feedback_log: list[str] = Field(default_factory=list)


@dataclass
class LinkedInToBlueskyDeps:
    brand: Brand
    settings: BlueskySettings  # reused bluesky_settings — no new settings table
    source_text: str
    target_count: int | None = None
    thread_preference: Literal["no_preference", "independent", "thread"] = "no_preference"
    recent_posts: list[str] = field(default_factory=list)  # recent_bluesky_texts(brand_id)
    feedback_log: list[str] = field(default_factory=list)
    # Set via dataclasses.replace before each per-group write call; unused
    # by the plan call itself. group_size/group_topic describe the one
    # group this particular write call is writing (see module docstring for
    # why it's one call per group, not one call for the whole plan).
    # written_so_far accumulates every earlier group's posts in this same
    # plan, purely so later groups can avoid repeating the same phrasing —
    # groups never depend on each other's *content* (see write_posts_agent's
    # instructions).
    group_size: int = 0
    group_topic: str = ""
    written_so_far: list[str] = field(default_factory=list)


def _shared_context_lines(deps: LinkedInToBlueskyDeps) -> list[str]:
    """Brand identity + Bluesky config + anti-repetition + feedback log —
    common to both calls below. recent_posts and feedback_log are given to
    both, not just the write call: avoiding repeated topics matters as much
    when deciding what to plan as when writing the actual copy.
    """
    brand = deps.brand
    lines = [
        f"Brand: {brand.name}",
        f"Background: {brand.background}" if brand.background else "",
        f"Voice/tone: {brand.voice}" if brand.voice else "",
        f"Audience: {brand.audience}" if brand.audience else "",
    ]
    if deps.settings.hashtags:
        lines.append(f"Hashtags to use: {deps.settings.hashtags}")
    if deps.settings.instructions:
        lines.append(f"Bluesky-specific instructions: {deps.settings.instructions}")
    if deps.recent_posts:
        recent = "\n".join(f"- {post}" for post in deps.recent_posts)
        lines.append(f"Recent Bluesky posts for this brand (do not repeat these):\n{recent}")
    if deps.feedback_log:
        feedback = "\n".join(f"- {note}" for note in deps.feedback_log)
        lines.append(
            f"Feedback/instructions given earlier in this session (all still apply):\n{feedback}"
        )
    return lines


# --- 1. Breakdown plan (internal only — never shown to the user) ---


class BreakdownPlanOutput(BaseModel):
    """Two parallel flat lists, not a `list[{size, topic}]` — the latter
    repeats the exact nested-object field-naming risk `HookOption`/
    `HashtagOption` already demonstrated (agent/agents/linkedin.py), which
    needed two rounds of prose tuning to stop the model substituting
    synonym keys for an even simpler 2-field object. Two flat lists are
    exactly as flat as the already-proven `BlueskyOutput.posts` shape.
    """

    group_sizes: list[Annotated[int, Field(ge=1, le=8)]] = Field(min_length=1, max_length=6)
    group_topics: list[str] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def _same_length(self) -> BreakdownPlanOutput:
        if len(self.group_sizes) != len(self.group_topics):
            raise ValueError(
                f"group_sizes has {len(self.group_sizes)} items but group_topics has "
                f"{len(self.group_topics)} — they must be the same length, one size and "
                "one topic per group, in the same order."
            )
        return self


breakdown_plan_agent: Agent[LinkedInToBlueskyDeps, BreakdownPlanOutput] = Agent(
    settings.model,
    name="linkedin_to_bluesky_plan_agent",  # labels this agent's run span in Logfire traces
    output_type=BreakdownPlanOutput,
    deps_type=LinkedInToBlueskyDeps,
    retries=_RETRIES,
    model_settings=_MODEL_SETTINGS,
    instructions="""You analyze a LinkedIn post and plan how to adapt it into one or more
Bluesky posts — you are not writing the posts yet, only deciding the breakdown.

Decide how to split the LinkedIn post's content into one or more groups. Each group
becomes either one standalone Bluesky post (a group of size 1, understandable on its
own with no other context) or a thread of sequential posts (a group of size greater
than 1, where each post continues only the post before it within that same group —
never depending on a different group).

Produce two lists, the same length and in the same order: group_sizes (each group's
post count) and group_topics (a short phrase describing what that group covers).

If a target total post count is given below, treat it as a soft guideline for the sum
of group_sizes, not a hard requirement. If a preference for independent posts vs. a
thread is given, treat it as a soft guideline too — freely mix independent posts and
threads regardless of the stated preference. When in doubt, let the shape of the
content decide, not the preference. If feedback from earlier regenerations is given,
apply it to this breakdown.""",
)


@breakdown_plan_agent.instructions
async def plan_context(ctx: RunContext[LinkedInToBlueskyDeps]) -> str:
    deps = ctx.deps
    lines = [*_shared_context_lines(deps), f"LinkedIn post to adapt:\n{deps.source_text}"]
    if deps.target_count:
        lines.append(f"Target total post count (soft guideline): {deps.target_count}")
    if deps.thread_preference != "no_preference":
        pref = "independent posts" if deps.thread_preference == "independent" else "a thread"
        lines.append(f"Preference (soft guideline): {pref}")
    return "\n".join(line for line in lines if line)


# --- 2. Write posts ---


class WritePostsOutput(BaseModel):
    """Deliberately identical shape to `BlueskyOutput.posts` — the flat
    schema already trusted against gemma4. One call writes one group, so
    this is just that group's posts in order — no re-chunking needed.
    """

    posts: list[Annotated[str, Field(max_length=300)]] = Field(min_length=1)


write_posts_agent: Agent[LinkedInToBlueskyDeps, WritePostsOutput] = Agent(
    settings.model,
    name="linkedin_to_bluesky_write_agent",  # labels this agent's run span in Logfire traces
    output_type=WritePostsOutput,
    deps_type=LinkedInToBlueskyDeps,
    retries=_RETRIES,
    model_settings=_MODEL_SETTINGS,
    instructions="""You write Bluesky posts, adapting a LinkedIn post per one group of an
already-decided breakdown plan. A separate call handles each other group in the plan —
you only write this one group's posts; do not write posts for any other group.

The group's post count and topic are given in the context below — return exactly that
many posts, in order, and nothing else.

If the group has only one post, it must be fully self-contained — understandable on its
own, with no reference to "the previous post" or anything outside itself. If the group
has more than one post, it's a thread: each post continues only the post before it
within this same group.

Every post must be 300 characters or fewer, including any hashtags. Ground every post
in the LinkedIn post's actual content — don't invent facts it doesn't contain. Write in
the brand's voice, for its audience.""",
)


@write_posts_agent.output_validator
async def validate_post_count(
    ctx: RunContext[LinkedInToBlueskyDeps], data: WritePostsOutput
) -> WritePostsOutput:
    """The expected count depends on ctx.deps.group_size — set fresh per
    call from the plan call's output, not anything encoded in
    WritePostsOutput's own schema — so this can't be a model_validator on
    the output model itself (contrast with BreakdownPlanOutput._same_length
    above, whose invariant IS fully self-contained in the model). ModelRetry
    draws from the same retries=3 budget as any other validation failure.

    This only has to match ONE group's size now, not the sum across every
    group in the plan (see module docstring) — a single small target number
    instead of several simultaneous ones, which is what actually made the
    original version's exact-count requirement unreliable against gemma4.
    """
    expected = ctx.deps.group_size
    if len(data.posts) != expected:
        raise ModelRetry(
            f"You returned {len(data.posts)} posts but this group ('{ctx.deps.group_topic}') "
            f"needs exactly {expected}. Return exactly {expected} posts in `posts`, nothing else."
        )
    return data


@write_posts_agent.instructions
async def write_context(ctx: RunContext[LinkedInToBlueskyDeps]) -> str:
    deps = ctx.deps
    lines = [
        *_shared_context_lines(deps),
        f"LinkedIn post to adapt:\n{deps.source_text}",
        f"This call writes exactly {deps.group_size} "
        f"post{'s' if deps.group_size != 1 else ''} for one group of the breakdown "
        f"plan: {deps.group_topic}",
    ]
    if deps.written_so_far:
        already = "\n".join(f"- {post}" for post in deps.written_so_far)
        lines.append(
            "Posts already written for other groups in this same plan (avoid repeating "
            f"their phrasing or framing):\n{already}"
        )
    return "\n".join(line for line in lines if line)


class PhaseCallback(Protocol):
    def __call__(self, phase: str) -> None: ...


async def run_linkedin_to_bluesky_agent(
    deps: LinkedInToBlueskyDeps,
    on_phase: PhaseCallback | None = None,
) -> list[LinkedInToBlueskyGroup]:
    """Plan the breakdown, then write each group's posts with its own call,
    sequentially, feeding each call what earlier groups already wrote (see
    module docstring for why it's one write call per group).

    Args:
        deps: Brand identity, reused Bluesky settings, the source LinkedIn
            text, target count / thread preference, anti-repetition
            context, and cumulative feedback.
        on_phase: Optional callback invoked with a human-readable phase name
            before the plan call and before each group's write call — see
            agent/agents/newsletter.py's identical parameter.
    """
    if on_phase:
        on_phase("Analyzing post and planning breakdown")
    logger.info("Running LinkedIn-to-Bluesky plan agent", extra={"brand_id": deps.brand.id})
    plan_result = await breakdown_plan_agent.run(
        deps.source_text, deps=deps, usage_limits=USAGE_LIMITS
    )
    group_sizes = plan_result.output.group_sizes
    group_topics = plan_result.output.group_topics
    # Only visible with AGENT_DEBUG=true (see agent/logging.py) — nothing
    # else captures the actual generated content on a *successful* call.
    logger.debug(
        "LinkedIn-to-Bluesky plan agent output",
        extra={"group_sizes": group_sizes, "group_topics": group_topics},
    )

    logger.info("Running LinkedIn-to-Bluesky write agent", extra={"brand_id": deps.brand.id})
    written_so_far: list[str] = []
    groups: list[LinkedInToBlueskyGroup] = []
    for i, (size, topic) in enumerate(zip(group_sizes, group_topics, strict=True)):
        if on_phase:
            phase = (
                "Writing posts"
                if len(group_sizes) == 1
                else f"Writing posts ({i + 1}/{len(group_sizes)})"
            )
            on_phase(phase)
        write_deps = replace(
            deps, group_size=size, group_topic=topic, written_so_far=written_so_far
        )
        write_result = await write_posts_agent.run(
            deps.source_text, deps=write_deps, usage_limits=USAGE_LIMITS
        )
        logger.debug(
            "LinkedIn-to-Bluesky write agent output",
            extra={"group_topic": topic, "posts": write_result.output.posts},
        )
        groups.append(LinkedInToBlueskyGroup(posts=write_result.output.posts))
        written_so_far = [*written_so_far, *write_result.output.posts]

    return groups
