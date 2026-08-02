"""Unit tests for the newsletter agent using TestModel (no real model calls —
see tests/conftest.py's autouse override).

Newsletter generation is split into two sequential calls (body, then
subjects informed by the generated body) — see agent/agents/newsletter.py.
"""

from agent.agents import NewsletterDeps, run_newsletter_agent
from agent.agents.newsletter import body_context, subjects_context
from db.models import Brand, NewsletterSettings


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


def _ctx(deps: NewsletterDeps) -> FakeCtx:
    ctx = FakeCtx()
    ctx.deps = deps
    return ctx


async def test_run_newsletter_agent_returns_five_subject_pairs():
    deps = NewsletterDeps(brand=_brand(), settings=NewsletterSettings(brand_id="brand-1"))
    output = await run_newsletter_agent("Write this month's update", deps)
    assert output.body_html
    assert len(output.subject_pairs) == 5


async def test_run_newsletter_agent_reports_both_phases_in_order():
    deps = NewsletterDeps(brand=_brand(), settings=NewsletterSettings(brand_id="brand-1"))
    phases: list[str] = []

    await run_newsletter_agent("Write this month's update", deps, on_phase=phases.append)

    assert phases == ["Generating newsletter body", "Generating subject lines"]


async def test_body_context_includes_instructions_and_html_template():
    deps = NewsletterDeps(
        brand=_brand(),
        settings=NewsletterSettings(
            brand_id="brand-1",
            instructions="3 sections, 200 words each",
            html_template="<html><body>{{content}}</body></html>",
        ),
    )
    instructions = await body_context(_ctx(deps))

    assert "3 sections, 200 words each" in instructions
    assert "{{content}}" in instructions
    assert "Acme" in instructions


async def test_body_context_includes_recent_summaries_for_anti_repetition():
    deps = NewsletterDeps(
        brand=_brand(),
        settings=NewsletterSettings(brand_id="brand-1"),
        recent_summaries=["Last month: widget sale announcement"],
    )
    instructions = await body_context(_ctx(deps))

    assert "Last month: widget sale announcement" in instructions


async def test_subjects_context_includes_generated_body_html():
    deps = NewsletterDeps(
        brand=_brand(),
        settings=NewsletterSettings(brand_id="brand-1"),
        body_html="<p>This month we launched a new widget.</p>",
    )
    instructions = await subjects_context(_ctx(deps))

    assert "This month we launched a new widget." in instructions
    assert "Acme" in instructions


async def test_subjects_context_does_not_include_html_template():
    """html_template is body-layout guidance — irrelevant once the body
    already exists, so only body_context should surface it.
    """
    deps = NewsletterDeps(
        brand=_brand(),
        settings=NewsletterSettings(
            brand_id="brand-1", html_template="<html><body>{{content}}</body></html>"
        ),
        body_html="<p>Body text</p>",
    )
    instructions = await subjects_context(_ctx(deps))

    assert "{{content}}" not in instructions
