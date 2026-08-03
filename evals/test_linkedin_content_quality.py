"""Content-quality evals for the LinkedIn wizard, against the real configured
model — targeting two specific failure modes found via a real debug-mode
walkthrough (AGENT_DEBUG=true, see agent/logging.py) rather than anticipated
ahead of time:

  - The outline step satisfied its old unconstrained `list[str]` schema with
    a single list item containing the *entire* post — headers, **bold**,
    nested sub-bullets, a closing CTA, and hashtags already baked in, instead
    of several short bullet points. The draft step then inherited that
    markdown formatting wholesale, and it took the user five rounds of manual
    "remove the markdown" / "you broke the fix from two turns ago" feedback
    to get a clean draft.
  - The draft step pulled a fact from `brand.background` that the user had
    deliberately cut from the accepted outline (a "Foundry VTT" bullet,
    verified byte-for-byte against the request log: present in
    `brand.background`, absent from the outline actually submitted, present
    again in the resulting draft).

Fixes: agent/agents/linkedin.py added `_NO_MARKDOWN_REMINDER` (outline/draft/
polish/hooks instructions), a min/max length constraint + explicit "each
bullet is its own short item" instruction on the outline schema
(`OutlineOutput`), and an explicit "the outline is the complete content
plan" instruction on the draft agent. Neither failure was a schema
*validation* failure — both outputs were technically valid against the old
schema — so unit tests with TestModel can't catch either; these evals catch
the actual generated content drifting back into either failure mode against
the real model.

Run with: uv run pytest -m eval
No API key needed against the default local Ollama model, but it does need
a real, running Ollama instance with the configured model pulled.
"""

from __future__ import annotations

import re

import pytest

from agent.agents import (
    LinkedInDeps,
    run_linkedin_draft_agent,
    run_linkedin_outline_agent,
    run_linkedin_polish_agent,
)
from db.models import Brand, LinkedInSettings

# Same reliability bar as evals/test_reliability.py, same reasoning: all N
# attempts are expected to pass, and an occasional failure is meaningful
# signal, not noise to raise N away.
ATTEMPTS = 5

# A single blob standing in for "the whole post," like the one that started
# this: headers, nested sub-bullets, a CTA, and hashtags already baked in.
# Real bullets from a correctly-shaped outline don't get anywhere near this.
_BLOB_BULLET_LENGTH_LIMIT = 300


def _brand() -> Brand:
    return Brand(
        id="eval-brand",
        name="Acme Telescopes",
        background="We sell handmade beginner telescope kits.",
        voice="nerdy and excited, a little playful",
        audience="amateur astronomers",
        skypilot_id="",
        created_at="2026-01-01T00:00:00+00:00",
    )


def _markdown_violations(text: str) -> list[str]:
    """Detects Markdown syntax LinkedIn won't render — the class of thing
    _NO_MARKDOWN_REMINDER exists to prevent. Not meant to be a bulletproof
    Markdown parser, just precise enough to catch a real regression: bold,
    italic, headers, and dash/asterisk list syntax, the same four patterns
    named in the reminder itself.
    """
    violations = []
    if "**" in text:
        violations.append("bold (**...**)")
    if re.search(r"(?<!\*)\*(?!\*)[^*\n]+(?<!\*)\*(?!\*)", text):
        violations.append("italic (*...*)")
    if re.search(r"(?m)^\s{0,3}#{1,6}\s", text):
        violations.append("markdown header (#...)")
    if re.search(r"(?m)^\s*[-*]\s", text):
        violations.append("markdown list item (-/* ...)")
    return violations


@pytest.mark.eval
async def test_linkedin_outline_bullets_are_short_and_markdown_free():
    """N/N outlines come back as several short, plain-text bullets — not one
    blob standing in for the whole post. Length is a heuristic proxy for
    "one bullet, one point" that the min/max length schema constraint alone
    can't guarantee (a model could still satisfy a 3-8 count with 3 giant
    items).
    """
    brand = _brand()
    deps = LinkedInDeps(
        brand=brand,
        settings=LinkedInSettings(brand_id=brand.id),
        concept="Announcing our new 70mm beginner telescope kit",
        accepted_hook="Ever wanted to see Saturn's rings from your backyard?",
        accepted_hashtags=["#Astronomy", "#Telescopes", "#Stargazing"],
    )

    failures: list[str] = []
    for i in range(ATTEMPTS):
        try:
            outline = await run_linkedin_outline_agent(deps)
        except Exception as e:  # noqa: BLE001 — deliberately broad: any failure counts
            failures.append(f"attempt {i + 1}: raised {e!r}")
            continue

        if len(outline) < 3:
            failures.append(f"attempt {i + 1}: only {len(outline)} bullet(s), want 3-8")
        for bullet in outline:
            if len(bullet) > _BLOB_BULLET_LENGTH_LIMIT:
                failures.append(
                    f"attempt {i + 1}: bullet is {len(bullet)} chars (>{_BLOB_BULLET_LENGTH_LIMIT}), "
                    f"looks like a whole-post blob, not a short point: {bullet[:120]!r}..."
                )
            markdown = _markdown_violations(bullet)
            if markdown:
                failures.append(f"attempt {i + 1}: bullet contains {markdown}: {bullet[:120]!r}...")

    assert not failures, (
        f"{len(failures)} failure(s) across {ATTEMPTS} outline attempts:\n" + "\n".join(failures)
    )


@pytest.mark.eval
async def test_linkedin_draft_is_markdown_free():
    """N/N drafts, built from a clean pre-set outline, come back with no
    Markdown syntax — the failure that previously took the user several
    regenerate rounds to fully strip out by hand.
    """
    brand = _brand()
    deps = LinkedInDeps(
        brand=brand,
        settings=LinkedInSettings(brand_id=brand.id),
        concept="Announcing our new 70mm beginner telescope kit",
        accepted_hook="Ever wanted to see Saturn's rings from your backyard?",
        accepted_hashtags=["#Astronomy", "#Telescopes", "#Stargazing"],
        outline_bullets=[
            "Introduce the new 70mm beginner telescope kit",
            "Highlight it ships fully assembled, no tools needed",
            "Mention the included star chart and setup guide",
            "Call to action: link in bio to order",
        ],
    )

    failures: list[str] = []
    for i in range(ATTEMPTS):
        try:
            draft = await run_linkedin_draft_agent(deps)
        except Exception as e:  # noqa: BLE001 — deliberately broad: any failure counts
            failures.append(f"attempt {i + 1}: raised {e!r}")
            continue

        markdown = _markdown_violations(draft)
        if markdown:
            failures.append(f"attempt {i + 1}: draft contains {markdown}")

    assert not failures, (
        f"{len(failures)}/{ATTEMPTS} draft attempts contained Markdown:\n" + "\n".join(failures)
    )


@pytest.mark.eval
async def test_linkedin_polish_output_is_markdown_free():
    """N/N polish passes over an already-clean draft don't introduce
    Markdown — polish_agent is instructed to actively strip it if present,
    but should also never add it to text that didn't have it.
    """
    brand = _brand()
    clean_draft = (
        "Ever wanted to see Saturn's rings from your backyard?\n\n"
        "Our new 70mm beginner telescope kit ships fully assembled, no tools "
        "needed, with a star chart and setup guide included.\n\n"
        "Link in bio to order.\n\n#Astronomy #Telescopes #Stargazing"
    )
    deps = LinkedInDeps(
        brand=brand,
        settings=LinkedInSettings(brand_id=brand.id),
        concept="Announcing our new 70mm beginner telescope kit",
        draft_text=clean_draft,
    )

    failures: list[str] = []
    for i in range(ATTEMPTS):
        try:
            polish = await run_linkedin_polish_agent(deps)
        except Exception as e:  # noqa: BLE001 — deliberately broad: any failure counts
            failures.append(f"attempt {i + 1}: raised {e!r}")
            continue

        markdown = _markdown_violations(polish.corrected_text)
        if markdown:
            failures.append(f"attempt {i + 1}: corrected_text contains {markdown}")

    assert not failures, (
        f"{len(failures)}/{ATTEMPTS} polish attempts introduced Markdown:\n" + "\n".join(failures)
    )


@pytest.mark.eval
async def test_linkedin_draft_stays_within_outline_scope():
    """N/N drafts don't introduce a fact from brand.background that the
    outline doesn't mention — the exact reproduction of the real failure:
    brand background here mentions a members-only retreat program
    ("Nebula Nights") that the outline deliberately omits, matching how the
    real brand's background mentioned a Foundry VTT launch that the user had
    cut from the outline before generating the draft.
    """
    brand = Brand(
        id="eval-brand-scope",
        name="Acme Telescopes",
        background=(
            "We sell handmade beginner telescope kits. We also run a "
            "members-only monthly stargazing retreat in the Mojave desert "
            "called Nebula Nights."
        ),
        voice="nerdy and excited, a little playful",
        audience="amateur astronomers",
        skypilot_id="",
        created_at="2026-01-01T00:00:00+00:00",
    )
    deps = LinkedInDeps(
        brand=brand,
        settings=LinkedInSettings(brand_id=brand.id),
        concept="Announcing our new 70mm beginner telescope kit",
        accepted_hook="Ever wanted to see Saturn's rings from your backyard?",
        accepted_hashtags=["#Astronomy", "#Telescopes", "#Stargazing"],
        outline_bullets=[
            "Introduce the new 70mm beginner telescope kit",
            "Highlight it ships fully assembled, no tools needed",
            "Mention the included star chart and setup guide",
            "Call to action: link in bio to order",
        ],
    )

    failures: list[str] = []
    for i in range(ATTEMPTS):
        try:
            draft = await run_linkedin_draft_agent(deps)
        except Exception as e:  # noqa: BLE001 — deliberately broad: any failure counts
            failures.append(f"attempt {i + 1}: raised {e!r}")
            continue

        if "nebula nights" in draft.lower():
            failures.append(
                f"attempt {i + 1}: draft mentions 'Nebula Nights', a background fact the "
                f"outline doesn't cover: {draft[:200]!r}..."
            )

    assert not failures, (
        f"{len(failures)}/{ATTEMPTS} draft attempts scope-crept beyond the outline:\n"
        + "\n".join(failures)
    )
