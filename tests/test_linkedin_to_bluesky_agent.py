"""Unit tests for the LinkedIn-to-Bluesky agent using TestModel (no real
model calls — see tests/conftest.py's autouse override).

Generation is a plan call followed by one write call per group — see
agent/agents/linkedin_to_bluesky.py's module docstring for why it's one
call per group rather than one call for the whole plan.
"""

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelRetry

from agent.agents.linkedin_to_bluesky import (
    BreakdownPlanOutput,
    LinkedInToBlueskyDeps,
    WritePostsOutput,
    plan_context,
    run_linkedin_to_bluesky_agent,
    validate_post_count,
    write_context,
)
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


class FakeCtx:
    pass


def _ctx(deps: LinkedInToBlueskyDeps) -> FakeCtx:
    ctx = FakeCtx()
    ctx.deps = deps
    return ctx


def _deps(**overrides) -> LinkedInToBlueskyDeps:
    defaults = dict(
        brand=_brand(),
        settings=BlueskySettings(brand_id="brand-1"),
        source_text="A LinkedIn post about our new product launch.",
    )
    defaults.update(overrides)
    return LinkedInToBlueskyDeps(**defaults)


# --- BreakdownPlanOutput's self-contained invariant ---


def test_breakdown_plan_output_accepts_matching_lengths():
    output = BreakdownPlanOutput(group_sizes=[1, 2], group_topics=["intro", "details"])
    assert output.group_sizes == [1, 2]


def test_breakdown_plan_output_rejects_mismatched_group_lists():
    with pytest.raises(ValidationError, match="must be the same length"):
        BreakdownPlanOutput(group_sizes=[1, 2], group_topics=["only one topic"])


# --- validate_post_count: depends on ctx.deps.group_size, not the output
# schema alone, so it's an @agent.output_validator, not a model_validator —
# test it directly rather than relying on an end-to-end run to exercise it
# (see test_run_linkedin_to_bluesky_agent_returns_groups_under_test_model
# below for why TestModel alone never triggers this path).


async def test_validate_post_count_accepts_matching_count():
    deps = _deps(group_size=2, group_topic="intro")
    data = WritePostsOutput(posts=["a", "b"])
    result = await validate_post_count(_ctx(deps), data)
    assert result is data


async def test_validate_post_count_rejects_mismatched_count():
    deps = _deps(group_size=3, group_topic="details")
    data = WritePostsOutput(posts=["a", "b"])  # group expects 3, model returned 2
    with pytest.raises(ModelRetry, match="needs exactly 3"):
        await validate_post_count(_ctx(deps), data)


# --- Context building ---


async def test_plan_context_includes_source_text_and_soft_guidelines():
    deps = _deps(
        source_text="Our team shipped a huge update this week.",
        target_count=5,
        thread_preference="thread",
    )
    instructions = await plan_context(_ctx(deps))

    assert "Our team shipped a huge update this week." in instructions
    assert "5" in instructions
    assert "a thread" in instructions
    assert "Acme" in instructions


async def test_plan_context_omits_guidelines_when_unset():
    deps = _deps(source_text="Plain post with no preferences set.")
    instructions = await plan_context(_ctx(deps))

    assert "Target total post count" not in instructions
    assert "Preference" not in instructions


async def test_write_context_describes_this_groups_size_and_topic():
    deps = _deps(
        source_text="Our team shipped a huge update this week.",
        group_size=2,
        group_topic="Behind-the-scenes details",
    )
    instructions = await write_context(_ctx(deps))

    assert "exactly 2 posts" in instructions
    assert "Behind-the-scenes details" in instructions


async def test_write_context_includes_earlier_groups_when_present():
    deps = _deps(
        group_size=1,
        group_topic="The wrap-up",
        written_so_far=["First post already written.", "Second post already written."],
    )
    instructions = await write_context(_ctx(deps))

    assert "First post already written." in instructions
    assert "Second post already written." in instructions


async def test_write_context_omits_earlier_groups_when_none_written_yet():
    deps = _deps(group_size=1, group_topic="The headline announcement")
    instructions = await write_context(_ctx(deps))

    assert "already written" not in instructions


# --- End-to-end smoke test ---


async def test_run_linkedin_to_bluesky_agent_returns_groups_under_test_model():
    """TestModel generates exactly min_length items for an unconstrained list
    and the `ge` minimum for an unconstrained int, so under the default
    autouse TestModel override this deterministically produces ONE group
    with ONE post — group_sizes=[1], group_topics=[<str>], and the one write
    call (for that single group) returns posts=[<str>], so
    len(posts) == group_size by construction and validate_post_count's
    ModelRetry path is never exercised here (see the dedicated tests above
    for that). This test only proves the plan call and the per-group write
    loop wire together correctly end to end.
    """
    output = await run_linkedin_to_bluesky_agent(_deps())
    assert len(output) == 1
    assert len(output[0].posts) == 1
    assert output[0].status == "pending"


async def test_run_linkedin_to_bluesky_agent_calls_on_phase_before_each_call():
    phases = []
    await run_linkedin_to_bluesky_agent(_deps(), on_phase=phases.append)
    assert phases == ["Analyzing post and planning breakdown", "Writing posts"]
