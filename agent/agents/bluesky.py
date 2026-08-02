"""Bluesky post/thread generation agent.

Built on the template's single-agent pattern (agent/agents/single.py, now
deleted) since generation needs no tool calls: "send to SkyPilot" is a
UI-triggered action after human review of the draft, not something this
agent decides to do — see integrations/skypilot.py, called from
web/routers/bluesky.py, not from a tool here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from agent.config import settings
from agent.logging import get_logger
from db.models import BlueskySettings, Brand

logger = get_logger(__name__)

# Guardrail against runaway agentic loops — see CLAUDE.md. A run that
# exceeds either limit raises UsageLimitExceeded instead of silently
# burning tokens.
USAGE_LIMITS = UsageLimits(request_limit=10, total_tokens_limit=100_000)


class BlueskyOutput(BaseModel):
    """A thread is more than one string in `posts`, in reply order.

    `posts` is a flat list of strings, not a list of `{text: ...}` objects —
    verified against the project's actual default model (gemma4 via Ollama):
    it reliably emits a flat string list but not a nested-object list, so
    the schema is kept as simple as the model can reliably satisfy (see
    CLAUDE.md's "keep tool interfaces simple" guidance, which applies to
    output schemas the same way).
    """

    posts: list[Annotated[str, Field(max_length=300)]] = Field(min_length=1)


@dataclass
class BlueskyDeps:
    brand: Brand
    settings: BlueskySettings
    recent_posts: list[str] = field(default_factory=list)
    want_thread: bool = False
    post_count: int | None = None


bluesky_agent: Agent[BlueskyDeps, BlueskyOutput] = Agent(
    settings.model,
    name="bluesky_agent",  # labels this agent's run span in Logfire traces
    output_type=BlueskyOutput,
    deps_type=BlueskyDeps,
    # pydantic-ai's default retries=1 gives the model exactly one chance to
    # self-correct a validation failure (e.g. going over 300 characters)
    # before hard-failing the whole generation. That's tight enough to matter
    # for a small local model like the project's gemma4 default, which trips
    # the length constraint occasionally — a few retries costs little against
    # USAGE_LIMITS (still well under request_limit) and turns an occasional
    # slip into a quiet self-correction instead of a user-visible error.
    retries=3,
    # `ollama show gemma4` reports this model's default temperature as 1.0 —
    # unusually high, and it measurably increases how often the model drifts
    # off the expected tool-call/structured-output format entirely rather
    # than just getting a field wrong. Lower is more reliable for structured
    # output while still leaving room for varied post copy.
    model_settings={"temperature": 0.6},
    instructions="""You write Bluesky posts for a brand.

Every post you write — each individual post if writing a thread — must be 300
characters or fewer, including any hashtags. Write in the brand's voice, for its
audience. Do not repeat jokes, phrasing, or topics from the brand's recent posts
listed in the context below; the goal is to keep content fresh.
""",
)


@bluesky_agent.instructions
async def brand_context(ctx: RunContext[BlueskyDeps]) -> str:
    """Assemble brand identity, Bluesky-specific config, and anti-repetition context."""
    deps = ctx.deps
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

    if deps.want_thread:
        count_hint = f" of about {deps.post_count} posts" if deps.post_count else ""
        lines.append(
            f"Generate a thread{count_hint}: return multiple posts in `posts`, each "
            "reading as a reply to the one before it, each still <=300 characters."
        )
    else:
        lines.append("Generate a single post: return exactly one item in `posts`.")

    if deps.recent_posts:
        recent = "\n".join(f"- {post}" for post in deps.recent_posts)
        lines.append(f"Recent posts for this brand (do not repeat these jokes/topics):\n{recent}")

    return "\n".join(line for line in lines if line)


async def run_bluesky_agent(prompt: str, deps: BlueskyDeps) -> BlueskyOutput:
    """Generate a Bluesky post or thread.

    Args:
        prompt: The user's request for what the post(s) should be about.
        deps: Brand identity, Bluesky settings, recent-post context, and
            thread preferences.
    """
    logger.info("Running Bluesky agent", extra={"brand_id": deps.brand.id, "prompt": prompt})
    result = await bluesky_agent.run(prompt, deps=deps, usage_limits=USAGE_LIMITS)
    logger.info("Bluesky agent run complete", extra={"post_count": len(result.output.posts)})
    return result.output
