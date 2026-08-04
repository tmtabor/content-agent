"""Content-quality evals for the LinkedIn-to-Bluesky agent — semantic
properties no schema can enforce: are independent (size-1) posts genuinely
self-contained, do thread (size>1) posts read as one coherent sequence, and
does no post depend on content that only appears in a *different* group?
These are exactly the two hard requirements agent/agents/linkedin_to_bluesky.py's
write instructions state explicitly ("fully self-contained... never
depending on a different group") but can't be checked with a regex
(contrast with evals/test_linkedin_content_quality.py's markdown checks,
which are regex-checkable) — hence the LLM-as-judge harness (evals/judge.py).

Run with: uv run pytest -m eval
Needs settings.judge_model configured (default anthropic:claude-opus-4-8,
see .env.example's AGENT_JUDGE_MODEL) in addition to the default local
Ollama model the agent itself runs against.
"""

from __future__ import annotations

import pytest

from agent.agents import LinkedInToBlueskyDeps, run_linkedin_to_bluesky_agent
from db.models import BlueskySettings, Brand
from evals.judge import judge_response


def _brand() -> Brand:
    return Brand(
        id="eval-brand-l2b",
        name="Acme Telescopes",
        background="We sell handmade beginner telescope kits.",
        voice="nerdy and excited, a little playful",
        audience="amateur astronomers",
        skypilot_id="",
        created_at="2026-01-01T00:00:00+00:00",
    )


# Deliberately covers several distinct topics (a product launch, a new
# support line, a community event, a thank-you to the team) — enough
# material that the plan call has a real reason to produce more than one
# group, giving these evals something to actually check.
_SOURCE_TEXT = (
    "Big week at Acme Telescopes! We just wrapped up our biggest product launch yet: the "
    "70mm StarFinder kit, designed for total beginners. It ships fully assembled, no tools "
    "needed, with a star chart and a setup guide included. We also opened a new customer "
    "support line staffed by real astronomers who can help you find your first target in "
    "the night sky. On the community side, we're kicking off a monthly virtual stargazing "
    "meetup where customers can share photos and ask questions live. It's been a wild few "
    "months of manufacturing hiccups, customer feedback sessions, and finally getting "
    "everything right. Huge thanks to our small but mighty team for making this happen."
)


async def _generate_groups():
    brand = _brand()
    deps = LinkedInToBlueskyDeps(
        brand=brand,
        settings=BlueskySettings(brand_id=brand.id, hashtags="#astronomy"),
        source_text=_SOURCE_TEXT,
    )
    return await run_linkedin_to_bluesky_agent(deps)


@pytest.mark.eval
async def test_independent_posts_are_self_contained():
    groups = await _generate_groups()
    independent = [group for group in groups if len(group.posts) == 1]
    if not independent:
        pytest.skip("this generation produced no independent (size-1) posts to check")

    failures = []
    for group in independent:
        post = group.posts[0]
        verdict = await judge_response(
            task="Write a standalone social media post.",
            response=post,
            criteria=(
                "The post must be fully understandable on its own, with no reference to "
                '"the previous post", "as mentioned above", or any other missing context '
                "from a different post. A reader seeing only this text should be able to "
                "follow it completely."
            ),
            threshold=0.7,
        )
        if not verdict.passed:
            failures.append(f"{post!r}: {verdict.reasoning}")

    assert not failures, "Independent post(s) not self-contained:\n" + "\n".join(failures)


@pytest.mark.eval
async def test_thread_posts_read_as_a_coherent_sequence():
    groups = await _generate_groups()
    threads = [group for group in groups if len(group.posts) > 1]
    if not threads:
        pytest.skip("this generation produced no thread (size>1) groups to check")

    failures = []
    for group in threads:
        joined = "\n---\n".join(group.posts)
        verdict = await judge_response(
            task="Write a multi-post social media thread, one post at a time.",
            response=joined,
            criteria=(
                "Each post (separated by ---) must read as a natural continuation of the "
                "post before it, forming one coherent thread about a single topic — not a "
                "disconnected list of unrelated posts."
            ),
            threshold=0.7,
        )
        if not verdict.passed:
            failures.append(f"{group.posts!r}: {verdict.reasoning}")

    assert not failures, "Thread(s) don't read as a coherent sequence:\n" + "\n".join(failures)


@pytest.mark.eval
async def test_posts_do_not_depend_on_other_groups_content():
    """Cross-group check: a group's posts should never depend on content
    that only appears in a *different* group — the other half of this
    agent's core requirement (see agent/agents/linkedin_to_bluesky.py's
    write instructions: "never depending on a different group").
    """
    groups = await _generate_groups()
    if len(groups) < 2:
        pytest.skip("this generation produced fewer than 2 groups — nothing to cross-check")

    failures = []
    for i, group in enumerate(groups):
        other_content = "\n".join(
            f"- {' '.join(other.posts)}" for j, other in enumerate(groups) if j != i
        )
        this_text = "\n".join(group.posts)
        verdict = await judge_response(
            task="Confirm a social media post/thread doesn't depend on other, separate posts.",
            response=this_text,
            criteria=(
                "This post/thread must be understandable without needing anything from the "
                "following OTHER, separate posts/threads (not part of this one):\n"
                f"{other_content}\n\n"
                "It's fine if topics are thematically related (same brand); it's only a "
                "failure if this text explicitly refers to or requires the other content to "
                'make sense (e.g. "as I mentioned in my last post").'
            ),
            threshold=0.7,
        )
        if not verdict.passed:
            failures.append(f"group {i}: {verdict.reasoning}")

    assert not failures, "Post(s) depend on a different group's content:\n" + "\n".join(failures)
