"""Newsletter generation, split into two smaller sequential agent calls.

Originally a single agent producing {body_html, subject_pairs} in one shot.
Against the project's actual default model (gemma4 via Ollama), that
combined schema occasionally failed with "Please return text or include
your response in a tool call" — the model drifting off the structured-
output format entirely on the more complex of the app's two schemas.
Splitting into a body-only call and a subjects-only call (each simpler than
the combined one) reduces how much any single call has to get right, the
same principle that fixed agent/agents/bluesky.py's post-schema issue.
It also lets subjects be written with the *actual* generated body in
context instead of both calls independently guessing from just the prompt,
which should make them more accurate, not just more reliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from agent.config import settings
from agent.logging import get_logger
from db.models import Brand, NewsletterSettings

logger = get_logger(__name__)

# Guardrail against runaway agentic loops — see CLAUDE.md. Each of the two
# calls below gets its own independent budget (they're separate top-level
# generations, not one delegating to the other, so there's no shared-usage
# tracking between them the way a supervisor/worker pair would use).
USAGE_LIMITS = UsageLimits(request_limit=10, total_tokens_limit=100_000)

# gemma4's default temperature (1.0, per `ollama show gemma4`) is unusually
# high and measurably increases how often the model drifts off the
# tool-call/structured-output format entirely rather than just getting a
# field wrong — retries alone don't reliably fix it if the model repeats the
# same mistake. Lower is more reliable for structured output while still
# leaving room for varied content. Shared by both calls below.
_MODEL_SETTINGS = {"temperature": 0.6}
# See the matching comment in agent/agents/bluesky.py — the default of 1
# gives the model exactly one chance to self-correct a validation failure
# before hard-failing the whole generation, which is tight for a small
# local model.
_RETRIES = 3


class SubjectPair(BaseModel):
    subject: str
    description: str


class NewsletterOutput(BaseModel):
    """The combined result callers see — assembled from the two calls below."""

    body_html: str
    subject_pairs: list[SubjectPair] = Field(
        min_length=5,
        max_length=5,
        description="Exactly 5 subject/description pairs for A/B testing.",
    )


class NewsletterBodyOutput(BaseModel):
    body_html: str


class NewsletterSubjectsOutput(BaseModel):
    subject_pairs: list[SubjectPair] = Field(
        min_length=5,
        max_length=5,
        description="Exactly 5 subject/description pairs for A/B testing.",
    )


@dataclass
class NewsletterDeps:
    brand: Brand
    settings: NewsletterSettings
    recent_summaries: list[str] = field(default_factory=list)
    # Set (via dataclasses.replace) before the subjects call, once the body
    # exists — unused by the body call, which is what generates it.
    body_html: str = ""


def _brand_context_lines(deps: NewsletterDeps) -> list[str]:
    """Shared brand-identity + newsletter-config context for both calls."""
    brand = deps.brand
    lines = [
        f"Brand: {brand.name}",
        f"Background: {brand.background}" if brand.background else "",
        f"Voice/tone: {brand.voice}" if brand.voice else "",
        f"Audience: {brand.audience}" if brand.audience else "",
    ]
    if deps.settings.instructions:
        lines.append(
            "Newsletter instructions (recurring subsections, word counts, subject-line "
            f"emoji, etc.): {deps.settings.instructions}"
        )
    if deps.recent_summaries:
        recent = "\n".join(f"- {summary}" for summary in deps.recent_summaries)
        lines.append(
            f"Recent newsletters for this brand (do not repeat these topics/angles):\n{recent}"
        )
    return lines


# --- Body agent ---

newsletter_body_agent: Agent[NewsletterDeps, NewsletterBodyOutput] = Agent(
    settings.model,
    name="newsletter_body_agent",  # labels this agent's run span in Logfire traces
    output_type=NewsletterBodyOutput,
    deps_type=NewsletterDeps,
    retries=_RETRIES,
    model_settings=_MODEL_SETTINGS,
    instructions="""You write the body of an email newsletter for a brand.

Produce the newsletter body as a single self-contained HTML fragment (body_html) —
valid HTML suitable for pasting into an email template, no <html>/<head>/<body> wrapper
needed. If an HTML template is given in the context below, follow its exact structure
and styling instead of inventing your own layout. Write in the brand's voice, for its
audience. Do not repeat topics or angles from the brand's recent newsletters listed in
the context below.
""",
)


@newsletter_body_agent.instructions
async def body_context(ctx: RunContext[NewsletterDeps]) -> str:
    deps = ctx.deps
    lines = _brand_context_lines(deps)
    if deps.settings.html_template:
        lines.append(
            "Always use this exact HTML template for body_html — fill in its content, "
            "keep its structure/styling, don't invent a different layout:\n"
            f"{deps.settings.html_template}"
        )
    return "\n".join(line for line in lines if line)


# --- Subject/description agent ---

newsletter_subjects_agent: Agent[NewsletterDeps, NewsletterSubjectsOutput] = Agent(
    settings.model,
    name="newsletter_subjects_agent",  # labels this agent's run span in Logfire traces
    output_type=NewsletterSubjectsOutput,
    deps_type=NewsletterDeps,
    retries=_RETRIES,
    model_settings=_MODEL_SETTINGS,
    instructions="""You write subject-line/preview-description pairs for an email newsletter.

You'll be given the newsletter's already-written HTML body in the context below — base
the subject lines and descriptions on what it actually says. Produce exactly 5 distinct
pairs suitable for A/B testing; each pair should take a genuinely different angle, not
minor rewordings of each other. Write in the brand's voice, for its audience. Do not
repeat subject lines or angles from the brand's recent newsletters listed in the context.
""",
)


@newsletter_subjects_agent.instructions
async def subjects_context(ctx: RunContext[NewsletterDeps]) -> str:
    deps = ctx.deps
    lines = _brand_context_lines(deps)
    lines.append(f"Newsletter body to write subject lines for:\n{deps.body_html}")
    return "\n".join(line for line in lines if line)


class PhaseCallback(Protocol):
    def __call__(self, phase: str) -> None: ...


async def run_newsletter_agent(
    prompt: str,
    deps: NewsletterDeps,
    on_phase: PhaseCallback | None = None,
) -> NewsletterOutput:
    """Generate a newsletter body, then 5 A/B subject/description pairs for it.

    Args:
        prompt: The user's request for what this newsletter should cover.
        deps: Brand identity, newsletter settings, and recent-content context.
        on_phase: Optional callback invoked with a human-readable phase name
            before each of the two calls — lets a caller (e.g. the web UI's
            job-polling layer) surface generation progress.
    """
    if on_phase:
        on_phase("Generating newsletter body")
    logger.info(
        "Running newsletter body agent", extra={"brand_id": deps.brand.id, "prompt": prompt}
    )
    body_result = await newsletter_body_agent.run(prompt, deps=deps, usage_limits=USAGE_LIMITS)
    body_html = body_result.output.body_html

    if on_phase:
        on_phase("Generating subject lines")
    logger.info("Running newsletter subjects agent", extra={"brand_id": deps.brand.id})
    subjects_deps = replace(deps, body_html=body_html)
    subjects_result = await newsletter_subjects_agent.run(
        prompt, deps=subjects_deps, usage_limits=USAGE_LIMITS
    )

    logger.info("Newsletter agent run complete")
    return NewsletterOutput(body_html=body_html, subject_pairs=subjects_result.output.subject_pairs)
