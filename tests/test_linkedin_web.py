"""End-to-end LinkedIn wizard tests via the shared `client` fixture (see
tests/conftest.py). The wizard's state is a JSON blob in a hidden `state`
form field — helpers below extract/decode it from responses (real HTML
attribute escaping and all) so tests can drive the wizard exactly the way
a browser submitting the rendered forms would.
"""

import html
import re
import time

from db.repository import create_brand, list_content_history


def _create_brand(**overrides):
    defaults = dict(
        name="Acme", background="Widgets", voice="playful", audience="makers", skypilot_id=""
    )
    defaults.update(overrides)
    return create_brand(**defaults)


def _extract_state(response_text: str) -> str:
    match = re.search(r'name="state" value="([^"]*)"', response_text)
    assert match, f"no state field found in:\n{response_text[:1000]}"
    return html.unescape(match.group(1))


def _poll(client, brand_id, response_text: str) -> str:
    """LinkedIn generation runs as a background job (see web/jobs.py) — a
    generate/regenerate endpoint returns a polling partial immediately, not
    the result. Poll /status until the job finishes (TestModel is
    effectively instant, so this resolves almost immediately).
    """
    match = re.search(r"/linkedin/status/([a-f0-9]+)", response_text)
    if not match:
        return response_text  # already a final render (cache hit, no job)
    job_id = match.group(1)
    for _ in range(50):
        response = client.get(f"/brands/{brand_id}/linkedin/status/{job_id}")
        if "generating-status" not in response.text:
            return response.text
        time.sleep(0.02)
    raise AssertionError("linkedin job did not finish within the polling budget")


def test_linkedin_page_renders_concept_form(client):
    brand = _create_brand()
    response = client.get(f"/brands/{brand.id}/linkedin")
    assert response.status_code == 200
    assert "Content idea" in response.text


def test_full_wizard_walkthrough_to_mark_used(client):
    brand = _create_brand()

    response = client.get(f"/brands/{brand.id}/linkedin")
    state = _extract_state(response.text)

    response = client.post(
        f"/brands/{brand.id}/linkedin/hooks/generate",
        data={"state": state, "concept": "Launching a new product"},
    )
    html_text = _poll(client, brand.id, response.text)
    assert "Choose a hook" in html_text

    state = _extract_state(html_text)
    response = client.post(
        f"/brands/{brand.id}/linkedin/hashtags/generate",
        data={"state": state, "selected_hook": "0", "hook_text_0": "My chosen hook"},
    )
    html_text = _poll(client, brand.id, response.text)
    assert "Choose hashtags" in html_text

    state = _extract_state(html_text)
    response = client.post(
        f"/brands/{brand.id}/linkedin/outline/generate",
        data={"state": state, "hashtag_0": "#test", "custom_hashtags": "#extra"},
    )
    html_text = _poll(client, brand.id, response.text)
    assert "Outline" in html_text

    state = _extract_state(html_text)
    response = client.post(
        f"/brands/{brand.id}/linkedin/draft/generate",
        data={"state": state, "outline_text": "point one\npoint two"},
    )
    html_text = _poll(client, brand.id, response.text)
    assert "Draft" in html_text

    state = _extract_state(html_text)
    response = client.post(
        f"/brands/{brand.id}/linkedin/polish/generate",
        data={"state": state, "draft_text": "my full post text"},
    )
    html_text = _poll(client, brand.id, response.text)
    assert "Final review" in html_text

    response = client.post(
        f"/brands/{brand.id}/linkedin/mark-used", data={"post_text": "final text"}
    )
    assert "Marked as used" in response.text

    history = list_content_history(brand.id, "linkedin")
    assert len(history) == 1
    assert history[0].payload.post_text == "final text"


def _reach_hashtags_step(client, brand_id):
    """Shared setup for the tests below: walk concept -> hooks -> hashtags,
    returning the hashtags step's HTML.
    """
    response = client.get(f"/brands/{brand_id}/linkedin")
    state = _extract_state(response.text)
    response = client.post(
        f"/brands/{brand_id}/linkedin/hooks/generate",
        data={"state": state, "concept": "Launching a new product"},
    )
    hooks_html = _poll(client, brand_id, response.text)

    state = _extract_state(hooks_html)
    response = client.post(
        f"/brands/{brand_id}/linkedin/hashtags/generate",
        data={"state": state, "selected_hook": "0", "hook_text_0": "My chosen hook"},
    )
    return hooks_html, _poll(client, brand_id, response.text)


def test_back_then_next_reuses_cached_content_no_new_job(client):
    """Going Back to hooks and then Next again (without regenerating)
    should reuse the cached hashtag_options — no second job, no polling
    partial in the response.
    """
    brand = _create_brand()
    hooks_html, hashtags_html = _reach_hashtags_step(client, brand.id)

    state = _extract_state(hashtags_html)
    back_response = client.post(f"/brands/{brand.id}/linkedin/back", data={"state": state})
    assert "Choose a hook" in back_response.text

    state = _extract_state(back_response.text)
    forward_response = client.post(
        f"/brands/{brand.id}/linkedin/hashtags/generate",
        data={"state": state, "selected_hook": "0", "hook_text_0": "My chosen hook"},
    )
    # A cache hit renders the hashtags partial directly — no polling partial.
    assert "generating-status" not in forward_response.text
    assert "Choose hashtags" in forward_response.text


def test_hooks_regenerate_clears_accepted_hook_and_dedupes(client):
    brand = _create_brand()
    response = client.get(f"/brands/{brand.id}/linkedin")
    state = _extract_state(response.text)
    response = client.post(
        f"/brands/{brand.id}/linkedin/hooks/generate",
        data={"state": state, "concept": "Launching a new product"},
    )
    hooks_html = _poll(client, brand.id, response.text)

    state = _extract_state(hooks_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin/hooks/regenerate",
        data={"state": state, "regen_instructions": "avoid focusing on price"},
    )
    regenerated_html = _poll(client, brand.id, response.text)
    assert "Choose a hook" in regenerated_html

    # accepted_hook was cleared by the regenerate — no radio should be
    # pre-selected against a now-superseded hook text.
    regenerated_state = _extract_state(regenerated_html)
    assert '"accepted_hook":""' in regenerated_state


def test_draft_to_polish_always_regenerates_even_with_cached_result(client):
    """The one deliberate exception to the caching rule: draft -> polish
    always starts a new job, even if a polish_result is already cached
    from an earlier visit (e.g. after Back then Next without editing).
    """
    brand = _create_brand()
    _, hashtags_html = _reach_hashtags_step(client, brand.id)
    state = _extract_state(hashtags_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin/outline/generate",
        data={"state": state, "hashtag_0": "#test", "custom_hashtags": ""},
    )
    outline_html = _poll(client, brand.id, response.text)

    state = _extract_state(outline_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin/draft/generate",
        data={"state": state, "outline_text": "point one"},
    )
    draft_html = _poll(client, brand.id, response.text)

    state = _extract_state(draft_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin/polish/generate",
        data={"state": state, "draft_text": "first draft text"},
    )
    polish_html = _poll(client, brand.id, response.text)
    assert "Final review" in polish_html

    # Back to draft, then Next again with the SAME draft text — a
    # polish_result is already cached in state, but this must still
    # trigger a fresh job (a polling partial in the immediate response).
    state = _extract_state(polish_html)
    back_response = client.post(f"/brands/{brand.id}/linkedin/back", data={"state": state})
    state = _extract_state(back_response.text)
    response = client.post(
        f"/brands/{brand.id}/linkedin/polish/generate",
        data={"state": state, "draft_text": "first draft text"},
    )
    assert "generating-status" in response.text


def test_mark_used_from_either_panel_saves_that_exact_text(client):
    brand = _create_brand()

    response1 = client.post(
        f"/brands/{brand.id}/linkedin/mark-used", data={"post_text": "original version"}
    )
    assert "Marked as used" in response1.text

    response2 = client.post(
        f"/brands/{brand.id}/linkedin/mark-used", data={"post_text": "corrected version"}
    )
    assert "Marked as used" in response2.text

    history = list_content_history(brand.id, "linkedin")
    assert {entry.payload.post_text for entry in history} == {
        "original version",
        "corrected version",
    }


def _reach_draft_step(client, brand_id):
    """Shared setup: walk all the way to the draft step, returning its HTML."""
    _, hashtags_html = _reach_hashtags_step(client, brand_id)
    state = _extract_state(hashtags_html)
    response = client.post(
        f"/brands/{brand_id}/linkedin/outline/generate",
        data={"state": state, "hashtag_0": "#test", "custom_hashtags": ""},
    )
    outline_html = _poll(client, brand_id, response.text)

    state = _extract_state(outline_html)
    response = client.post(
        f"/brands/{brand_id}/linkedin/draft/generate",
        data={"state": state, "outline_text": "point one"},
    )
    return _poll(client, brand_id, response.text)


def test_regenerate_outline_captures_edited_text_into_state(client, monkeypatch):
    """Previously outline/regenerate only read `state`/`feedback`, silently
    dropping whatever the user had typed into the outline textarea, and
    outline_context never showed the model the current outline at all —
    directly assert the deps passed to the agent contain the edited text,
    the concrete regression check for the bug.
    """
    brand = _create_brand()
    _, hashtags_html = _reach_hashtags_step(client, brand.id)
    state = _extract_state(hashtags_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin/outline/generate",
        data={"state": state, "hashtag_0": "#test", "custom_hashtags": ""},
    )
    outline_html = _poll(client, brand.id, response.text)

    captured_deps = []
    import agent.agents.linkedin as linkedin_module

    original = linkedin_module.run_linkedin_outline_agent

    async def spy(deps):
        captured_deps.append(deps)
        return await original(deps)

    monkeypatch.setattr("web.routers.linkedin.run_linkedin_outline_agent", spy)

    state = _extract_state(outline_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin/outline/regenerate",
        data={
            "state": state,
            "outline_text": "my hand-edited point",
            "feedback": "make it punchier",
        },
    )
    _poll(client, brand.id, response.text)

    assert len(captured_deps) == 1
    assert captured_deps[0].outline_bullets == ["my hand-edited point"]
    assert "make it punchier" in captured_deps[0].feedback_log


def test_regenerate_draft_captures_edited_text_into_state(client, monkeypatch):
    """Same as above for the draft step, and directly asserts the deps
    passed to the agent actually contain the edited draft_text — the
    concrete regression check for the bug (regenerate silently dropping
    edits and never showing the model what it's revising).
    """
    brand = _create_brand()
    draft_html = _reach_draft_step(client, brand.id)

    captured_deps = []
    import agent.agents.linkedin as linkedin_module

    original = linkedin_module.run_linkedin_draft_agent

    async def spy(deps):
        captured_deps.append(deps)
        return await original(deps)

    monkeypatch.setattr("web.routers.linkedin.run_linkedin_draft_agent", spy)

    state = _extract_state(draft_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin/draft/regenerate",
        data={
            "state": state,
            "draft_text": "my hand-edited draft text",
            "feedback": "make it punchier",
        },
    )
    _poll(client, brand.id, response.text)

    assert len(captured_deps) == 1
    assert captured_deps[0].draft_text == "my hand-edited draft text"
    assert "make it punchier" in captured_deps[0].feedback_log
