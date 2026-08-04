"""Reliability evals: does generation actually succeed, repeatedly, against
the real configured model — and does it respect the hard constraints the
schema can't fully enforce on its own (Bluesky's 300-char cap; newsletter's
5 distinct, non-duplicate subject/description pairs)?

Why this exists: building this app's default model (ollama:gemma4, a small
local model — see .env.example) turned up two real failures that unit tests
with TestModel can never catch, because TestModel always emits schema-
conforming stub data on the first attempt:

  - Bluesky: occasionally exceeded 300 characters (ValidationError,
    string_too_long).
  - Newsletter: occasionally returned neither valid text nor a valid tool
    call at all (ToolRetryError), especially on the combined body+subjects
    schema — which is why newsletter generation is now two smaller
    sequential calls (see agent/agents/newsletter.py) instead of one.

Fixes applied: retries=3 (up from pydantic-ai's default of 1) and a lower
temperature (0.6, down from gemma4's default of 1.0) on every agent, plus
the newsletter split above. Manual verification after those fixes reached
5/5 on both agents. These evals turn that one-off manual check into a
repeatable one — if a future model swap, prompt edit, or schema change
regresses reliability, this is what will catch it before a user does.

Run with: uv run pytest -m eval
No API key needed against the default local Ollama model, but it does need
a real, running Ollama instance with the configured model pulled — these
are real generations, not against TestModel, and each newsletter run is two
sequential model calls, so the full file takes real time to run.
"""

from __future__ import annotations

import pytest

from agent.agents import (
    BlueskyDeps,
    LinkedInDeps,
    LinkedInToBlueskyDeps,
    NewsletterDeps,
    run_bluesky_agent,
    run_linkedin_draft_agent,
    run_linkedin_hashtags_agent,
    run_linkedin_hooks_agent,
    run_linkedin_outline_agent,
    run_linkedin_polish_agent,
    run_linkedin_to_bluesky_agent,
    run_newsletter_agent,
)
from db.models import BlueskySettings, Brand, LinkedInSettings, NewsletterSettings

# Matches the reliability bar this was tuned against — see module docstring.
# All N attempts are expected to succeed; an occasional failure here is
# meaningful signal that reliability regressed, not just noise to raise N
# away. Keep this deliberately strict rather than loosening it to reduce
# flakiness.
ATTEMPTS = 5


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


@pytest.mark.eval
async def test_bluesky_generation_reliability():
    """N/N single-post generations succeed and respect the 300-char cap."""
    brand = _brand()
    deps = BlueskyDeps(
        brand=brand,
        settings=BlueskySettings(brand_id=brand.id, hashtags="#ttrpgsky"),
    )

    failures: list[str] = []
    for i in range(ATTEMPTS):
        try:
            output = await run_bluesky_agent(f"Announce a 20% off sale, attempt {i + 1}", deps)
        except Exception as e:  # noqa: BLE001 — deliberately broad: any failure counts
            failures.append(f"attempt {i + 1}: raised {e!r}")
            continue

        if not output.posts:
            failures.append(f"attempt {i + 1}: no posts returned")
            continue
        for post in output.posts:
            if len(post) > 300:
                failures.append(f"attempt {i + 1}: post exceeded 300 chars ({len(post)})")

    assert not failures, (
        f"{len(failures)}/{ATTEMPTS} Bluesky generation attempts failed:\n" + "\n".join(failures)
    )


@pytest.mark.eval
async def test_newsletter_generation_reliability():
    """N/N generations succeed with a body and exactly 5 distinct subject pairs."""
    brand = _brand()
    deps = NewsletterDeps(
        brand=brand,
        settings=NewsletterSettings(
            brand_id=brand.id, instructions="Keep it to two short sections."
        ),
    )

    failures: list[str] = []
    for i in range(ATTEMPTS):
        try:
            output = await run_newsletter_agent(f"A quick weekly update, attempt {i + 1}", deps)
        except Exception as e:  # noqa: BLE001 — deliberately broad: any failure counts
            failures.append(f"attempt {i + 1}: raised {e!r}")
            continue

        if not output.body_html.strip():
            failures.append(f"attempt {i + 1}: empty body_html")
        if len(output.subject_pairs) != 5:
            failures.append(f"attempt {i + 1}: {len(output.subject_pairs)} subject pairs, want 5")
        subjects = [pair.subject for pair in output.subject_pairs]
        if len(set(subjects)) != len(subjects):
            failures.append(f"attempt {i + 1}: duplicate subject lines: {subjects}")

    assert not failures, (
        f"{len(failures)}/{ATTEMPTS} newsletter generation attempts failed:\n" + "\n".join(failures)
    )


@pytest.mark.eval
async def test_linkedin_generation_reliability():
    """N/N full wizard pipelines (hooks -> hashtags -> outline -> draft ->
    polish) succeed with correctly-shaped output at every step, each step
    informed by the real output of the one before it — not synthetic,
    isolated inputs. `polish` is the schema worth watching most closely
    (see agent/agents/linkedin.py's module docstring: a string field plus a
    list-of-strings field is structurally similar to the original combined
    NewsletterOutput that turned out unreliable), so this exercises it in
    the actual multi-step context it will really run in, not alone.
    """
    brand = _brand()
    settings = LinkedInSettings(brand_id=brand.id, instructions="Keep a friendly, expert tone.")

    failures: list[str] = []
    for i in range(ATTEMPTS):
        deps = LinkedInDeps(
            brand=brand, settings=settings, concept=f"Launching a new product, attempt {i + 1}"
        )
        try:
            hooks = await run_linkedin_hooks_agent(deps)
            if len(hooks) != 5:
                failures.append(f"attempt {i + 1}: {len(hooks)} hooks, want 5")
                continue
            deps.accepted_hook = hooks[0].hook_text

            hashtags = await run_linkedin_hashtags_agent(deps)
            if len(hashtags) != 10:
                failures.append(f"attempt {i + 1}: {len(hashtags)} hashtags, want 10")
                continue
            deps.accepted_hashtags = [tag.hashtag for tag in hashtags[:3]]

            outline = await run_linkedin_outline_agent(deps)
            if not outline:
                failures.append(f"attempt {i + 1}: empty outline")
                continue
            deps.outline_bullets = outline

            draft = await run_linkedin_draft_agent(deps)
            if not draft.strip():
                failures.append(f"attempt {i + 1}: empty draft")
                continue
            deps.draft_text = draft

            polish = await run_linkedin_polish_agent(deps)
            if not polish.corrected_text.strip():
                failures.append(f"attempt {i + 1}: empty polish.corrected_text")
        except Exception as e:  # noqa: BLE001 — deliberately broad: any failure counts
            failures.append(f"attempt {i + 1}: raised {e!r}")

    assert not failures, (
        f"{len(failures)}/{ATTEMPTS} LinkedIn generation attempts failed:\n" + "\n".join(failures)
    )


@pytest.mark.eval
async def test_linkedin_to_bluesky_generation_reliability():
    """N/N runs of the plan-then-write pipeline succeed with every post
    respecting the 300-char cap.

    The invariant unique to this agent — total posts always equals
    sum(group_sizes) — isn't asserted directly here because it doesn't need
    to be: agent/agents/linkedin_to_bluesky.py's validate_post_count raises
    ModelRetry on a mismatch, and pydantic-ai turns an unresolved ModelRetry
    into a hard failure once retries are exhausted. So a run reaching this
    point without raising is itself proof the counts converged — this eval
    is what actually exercises that ModelRetry path against the real model
    (see tests/test_linkedin_to_bluesky_agent.py for the direct, isolated
    unit test of validate_post_count, which TestModel can't exercise
    end-to-end).
    """
    brand = _brand()
    settings = BlueskySettings(brand_id=brand.id, hashtags="#ttrpgsky")
    source_text = (
        "Excited to share our team just shipped a major platform update! Over the past "
        "three months we rebuilt onboarding from the ground up, cutting time-to-first-value "
        "from 20 minutes to under 3. We also added a real-time analytics dashboard. None of "
        "this would have been possible without an incredible team. Check out the new "
        "dashboard today — we'd love your feedback!"
    )

    failures: list[str] = []
    for i in range(ATTEMPTS):
        deps = LinkedInToBlueskyDeps(
            brand=brand, settings=settings, source_text=source_text, target_count=4
        )
        try:
            groups = await run_linkedin_to_bluesky_agent(deps)
        except Exception as e:  # noqa: BLE001 — deliberately broad: any failure counts
            failures.append(f"attempt {i + 1}: raised {e!r}")
            continue

        if not groups:
            failures.append(f"attempt {i + 1}: no groups returned")
            continue
        for group in groups:
            for post in group.posts:
                if len(post) > 300:
                    failures.append(f"attempt {i + 1}: post exceeded 300 chars ({len(post)})")

    assert not failures, (
        f"{len(failures)}/{ATTEMPTS} LinkedIn-to-Bluesky generation attempts failed:\n"
        + "\n".join(failures)
    )
