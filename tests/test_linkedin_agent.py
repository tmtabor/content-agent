"""Unit tests for the LinkedIn agents using TestModel (no real model calls —
see tests/conftest.py's autouse override).

LinkedIn generation is five sequential agents (hooks, hashtags, outline,
draft, polish) sharing one LinkedInDeps — see agent/agents/linkedin.py.
"""

from agent.agents.linkedin import (
    LinkedInDeps,
    draft_context,
    hashtags_context,
    hooks_context,
    outline_context,
    polish_context,
    run_linkedin_draft_agent,
    run_linkedin_hashtags_agent,
    run_linkedin_hooks_agent,
    run_linkedin_outline_agent,
    run_linkedin_polish_agent,
)
from db.models import Brand, LinkedInSettings


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


def _ctx(deps: LinkedInDeps) -> FakeCtx:
    ctx = FakeCtx()
    ctx.deps = deps
    return ctx


def _deps(**overrides) -> LinkedInDeps:
    defaults = dict(brand=_brand(), settings=LinkedInSettings(brand_id="brand-1"))
    defaults.update(overrides)
    return LinkedInDeps(**defaults)


# --- Each agent returns schema-conforming output ---


async def test_run_hooks_agent_returns_five_options():
    output = await run_linkedin_hooks_agent(_deps(concept="Launching a new product"))
    assert len(output) == 5
    assert all(hook.hook_text and hook.explanation for hook in output)


async def test_run_hashtags_agent_returns_ten_options():
    output = await run_linkedin_hashtags_agent(_deps(concept="Launching a new product"))
    assert len(output) == 10
    assert all(tag.hashtag and tag.description for tag in output)


async def test_run_outline_agent_returns_bullet_list():
    output = await run_linkedin_outline_agent(_deps(concept="Launching a new product"))
    assert isinstance(output, list)
    assert all(isinstance(bullet, str) for bullet in output)


async def test_run_draft_agent_returns_plain_string():
    output = await run_linkedin_draft_agent(_deps(concept="Launching a new product"))
    assert isinstance(output, str)
    assert output


async def test_run_polish_agent_returns_corrected_text_and_changes():
    output = await run_linkedin_polish_agent(_deps(draft_text="my drafted post"))
    assert output.corrected_text
    assert isinstance(output.changes, list)


# --- Context building ---


async def test_hooks_context_includes_concept_and_dedupe_and_regen_instructions():
    deps = _deps(
        concept="Launching a new product",
        previous_options=["Old hook one", "Old hook two"],
        regen_instructions="avoid focusing on price",
    )
    instructions = await hooks_context(_ctx(deps))

    assert "Launching a new product" in instructions
    assert "Old hook one" in instructions
    assert "avoid focusing on price" in instructions
    assert "Acme" in instructions


async def test_hashtags_context_includes_chosen_hook_and_dedupe():
    deps = _deps(
        concept="Launching a new product",
        accepted_hook="My chosen hook",
        previous_options=["#oldtag"],
    )
    instructions = await hashtags_context(_ctx(deps))

    assert "My chosen hook" in instructions
    assert "#oldtag" in instructions


async def test_outline_context_includes_hook_and_hashtags():
    deps = _deps(
        concept="Launching a new product",
        accepted_hook="My chosen hook",
        accepted_hashtags=["#tag1", "#tag2"],
    )
    instructions = await outline_context(_ctx(deps))

    assert "My chosen hook" in instructions
    assert "#tag1" in instructions
    assert "#tag2" in instructions


async def test_draft_context_includes_outline_bullets():
    deps = _deps(
        concept="Launching a new product",
        accepted_hook="My chosen hook",
        accepted_hashtags=["#tag1"],
        outline_bullets=["First point", "Second point"],
    )
    instructions = await draft_context(_ctx(deps))

    assert "First point" in instructions
    assert "Second point" in instructions


# --- Regenerate must show the model what it's revising, not just a feedback
# note — this is the bug the reliability eval couldn't catch (it's not a
# validation failure) but a real user session did: regenerating the draft 7
# times in a row without converging, because outline_context/draft_context
# used to omit the current outline/draft entirely, making "regenerate with
# feedback" a blind rewrite from scratch every time instead of a revision.


async def test_outline_context_includes_current_outline_to_revise_when_present():
    deps = _deps(
        concept="Launching a new product",
        accepted_hook="My chosen hook",
        outline_bullets=["Old point one", "Old point two"],
        feedback_log=["make it shorter"],
    )
    instructions = await outline_context(_ctx(deps))

    assert "Old point one" in instructions
    assert "revise" in instructions.lower()


async def test_outline_context_omits_revise_note_on_fresh_generation():
    """No current outline yet (first generation, not a regenerate) — nothing
    to revise, so the agent shouldn't be told it's revising something.
    """
    deps = _deps(concept="Launching a new product", accepted_hook="My chosen hook")
    instructions = await outline_context(_ctx(deps))

    assert "revise" not in instructions.lower()


async def test_draft_context_includes_current_draft_to_revise_when_present():
    deps = _deps(
        concept="Launching a new product",
        accepted_hook="My chosen hook",
        outline_bullets=["First point"],
        draft_text="The old drafted post text.",
        feedback_log=["make the tone more casual"],
    )
    instructions = await draft_context(_ctx(deps))

    assert "The old drafted post text." in instructions
    assert "revise" in instructions.lower()


async def test_draft_context_omits_revise_note_on_fresh_generation():
    deps = _deps(
        concept="Launching a new product",
        accepted_hook="My chosen hook",
        outline_bullets=["First point"],
    )
    instructions = await draft_context(_ctx(deps))

    assert "revise" not in instructions.lower()


async def test_polish_context_includes_draft_text():
    deps = _deps(draft_text="the full drafted post")
    instructions = await polish_context(_ctx(deps))
    assert "the full drafted post" in instructions


# --- Feedback log: the spec's "retains in its context every accepted step
# and feedback" applies to feedback too, not just accepted content, so it
# must still be visible in a step several stages after the one it was
# entered at — the easiest part of this to under-build.


async def test_feedback_from_an_earlier_step_reaches_a_later_step():
    deps = _deps(
        concept="Launching a new product",
        accepted_hook="My chosen hook",
        accepted_hashtags=["#tag1"],
        outline_bullets=["First point"],
        feedback_log=["avoid focusing on price"],  # e.g. entered during hook regeneration
    )
    draft_instructions = await draft_context(_ctx(deps))
    polish_instructions = await polish_context(_ctx(deps))

    assert "avoid focusing on price" in draft_instructions
    assert "avoid focusing on price" in polish_instructions
