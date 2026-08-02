"""Unit tests for the Bluesky agent using TestModel (no real model calls —
see tests/conftest.py's autouse override).
"""

from agent.agents import BlueskyDeps, run_bluesky_agent
from agent.agents.bluesky import brand_context
from db.models import BlueskySettings, Brand


def _brand() -> Brand:
    return Brand(
        id="brand-1",
        name="Acme",
        background="Widgets for hobbyists",
        voice="playful",
        audience="makers",
        skypilot_id="sky-1",
        created_at="2026-01-01T00:00:00+00:00",
    )


async def test_run_bluesky_agent_returns_output():
    deps = BlueskyDeps(brand=_brand(), settings=BlueskySettings(brand_id="brand-1"))
    output = await run_bluesky_agent("Announce our new gadget", deps)
    assert output.posts
    assert all(output.posts)


async def test_brand_context_includes_hashtags_and_instructions():
    deps = BlueskyDeps(
        brand=_brand(),
        settings=BlueskySettings(
            brand_id="brand-1", instructions="No emojis", hashtags="#ttrpgsky"
        ),
    )

    class FakeCtx:
        pass

    ctx = FakeCtx()
    ctx.deps = deps
    instructions = await brand_context(ctx)

    assert "#ttrpgsky" in instructions
    assert "No emojis" in instructions
    assert "Acme" in instructions


async def test_brand_context_single_post_by_default():
    deps = BlueskyDeps(brand=_brand(), settings=BlueskySettings(brand_id="brand-1"))

    class FakeCtx:
        pass

    ctx = FakeCtx()
    ctx.deps = deps
    instructions = await brand_context(ctx)

    assert "single post" in instructions.lower()


async def test_brand_context_thread_mentions_post_count():
    deps = BlueskyDeps(
        brand=_brand(),
        settings=BlueskySettings(brand_id="brand-1"),
        want_thread=True,
        post_count=4,
    )

    class FakeCtx:
        pass

    ctx = FakeCtx()
    ctx.deps = deps
    instructions = await brand_context(ctx)

    assert "thread" in instructions.lower()
    assert "4" in instructions


async def test_brand_context_includes_recent_posts_for_anti_repetition():
    deps = BlueskyDeps(
        brand=_brand(),
        settings=BlueskySettings(brand_id="brand-1"),
        recent_posts=["Old joke about widgets", "Another old post"],
    )

    class FakeCtx:
        pass

    ctx = FakeCtx()
    ctx.deps = deps
    instructions = await brand_context(ctx)

    assert "Old joke about widgets" in instructions
    assert "Another old post" in instructions
