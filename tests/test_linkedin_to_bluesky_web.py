"""End-to-end LinkedIn-to-Bluesky tests via the shared `client` fixture (see
tests/conftest.py). Like tests/test_linkedin_web.py, the wizard's state is a
JSON blob in a hidden `state` form field — helpers below extract/decode it
from responses so tests can drive the flow exactly the way a browser
submitting the rendered forms would.

Under the default autouse TestModel override, every generate/regenerate
deterministically produces exactly one pending group with exactly one post
(see tests/test_linkedin_to_bluesky_agent.py's
test_run_linkedin_to_bluesky_agent_returns_groups_under_test_model for why)
— tests below rely on that to keep assertions concrete.
"""

import html
import json
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


def _pending_post_fields(state_json: str) -> dict[str, str]:
    """Reconstructs the post_{gi}_{pi} fields a browser would submit for
    every pending group's textareas — needed on every mutation endpoint so
    _sync_pending_edits doesn't blank out posts the test isn't trying to
    change (see web/routers/linkedin_to_bluesky.py's _sync_pending_edits).
    """
    parsed = json.loads(state_json)
    fields = {}
    for gi, group in enumerate(parsed["groups"]):
        if group["status"] != "pending":
            continue
        for pi, text in enumerate(group["posts"]):
            fields[f"post_{gi}_{pi}"] = text
    return fields


def _group_ids(response_text: str) -> list[str]:
    """Pending group ids, in document order (every pending group renders a
    Mark as Used button).
    """
    return re.findall(r"/groups/([a-f0-9]+)/mark-used", response_text)


def _poll(client, brand_id, response_text: str) -> str:
    match = re.search(r"/linkedin-to-bluesky/status/([a-f0-9]+)", response_text)
    if not match:
        return response_text  # already a final render
    job_id = match.group(1)
    for _ in range(50):
        response = client.get(f"/brands/{brand_id}/linkedin-to-bluesky/status/{job_id}")
        if "generating-status" not in response.text:
            return response.text
        time.sleep(0.02)
    raise AssertionError("linkedin-to-bluesky job did not finish within the polling budget")


def _generate(client, brand_id, **form_overrides) -> str:
    data = {"source_text": "A LinkedIn post about our new product launch."}
    data.update(form_overrides)
    response = client.post(f"/brands/{brand_id}/linkedin-to-bluesky/generate", data=data)
    return _poll(client, brand_id, response.text)


def test_linkedin_to_bluesky_page_renders_input_form(client):
    brand = _create_brand()
    response = client.get(f"/brands/{brand.id}/linkedin-to-bluesky")
    assert response.status_code == 200
    assert "LinkedIn post text" in response.text


def test_from_linkedin_handoff_prefills_source_text(client):
    brand = _create_brand()
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/from-linkedin",
        data={"source_text": "A post written on the LinkedIn wizard's Polish step."},
    )
    assert response.status_code == 200
    assert "A post written on the LinkedIn wizard&#39;s Polish step." in response.text


def test_from_linkedin_handoff_records_linkedin_history(client):
    """A user who adapts a post to Bluesky right from the Polish step never
    clicks LinkedIn's own "Mark as Used" — without this, the post it was
    built from never showed up in LinkedIn history at all, even though
    handing it off is itself a clear signal the user is using it.
    """
    brand = _create_brand()
    client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/from-linkedin",
        data={"source_text": "A post written on the LinkedIn wizard's Polish step."},
    )
    entries = list_content_history(brand.id, "linkedin")
    assert len(entries) == 1
    assert entries[0].payload.post_text == "A post written on the LinkedIn wizard's Polish step."


def test_from_linkedin_handoff_skips_history_for_blank_source_text(client):
    brand = _create_brand()
    client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/from-linkedin",
        data={"source_text": "   "},
    )
    assert list_content_history(brand.id, "linkedin") == []


def test_generate_then_poll_returns_one_pending_group(client):
    brand = _create_brand()
    results_html = _generate(client, brand.id)
    assert "group-card" in results_html
    assert "status-badge" not in results_html  # nothing sent/used yet

    state = json.loads(_extract_state(results_html))
    assert len(state["groups"]) == 1
    assert state["groups"][0]["status"] == "pending"


def test_send_group_persists_to_bluesky_history_and_shows_sent_badge(client, monkeypatch):
    brand = _create_brand(skypilot_id="sky-abc")
    results_html = _generate(client, brand.id)

    captured = {}

    async def fake_create_post(account_id, texts, scheduled_for=None):
        captured["account_id"] = account_id
        captured["texts"] = texts
        return {"id": "post-999"}

    monkeypatch.setattr("web.routers.linkedin_to_bluesky.create_post", fake_create_post)

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/send",
        data={"state": state, **_pending_post_fields(state)},
    )
    assert response.status_code == 200
    assert "Sent" in response.text
    assert captured["account_id"] == "sky-abc"

    history = list_content_history(brand.id, "bluesky")
    assert len(history) == 1
    assert history[0].skypilot_post_id == "post-999"


def test_send_group_without_skypilot_id_shows_error(client):
    brand = _create_brand(skypilot_id="")
    results_html = _generate(client, brand.id)

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/send",
        data={"state": state, **_pending_post_fields(state)},
    )
    assert "no SkyPilot account ID" in response.text
    assert list_content_history(brand.id, "bluesky") == []


def test_mark_used_group_persists_edited_text_and_shows_used_badge(client):
    brand = _create_brand()
    results_html = _generate(client, brand.id)

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/mark-used",
        data={"state": state, "post_0_0": "hand-edited final text"},
    )
    assert "Used" in response.text

    history = list_content_history(brand.id, "bluesky")
    assert len(history) == 1
    assert history[0].payload.posts[0].text == "hand-edited final text"


def test_send_group_with_overlong_edited_post_shows_friendly_error(client, monkeypatch):
    """A user can freely edit post text in a plain <textarea> with no
    client-side length limit — sending must reject an over-300-char edit
    with a clear message, not by letting BlueskyPost's own Field(max_length)
    raise unhandled inside add_content_history.
    """
    brand = _create_brand(skypilot_id="sky-abc")
    results_html = _generate(client, brand.id)

    async def fake_create_post(account_id, texts, scheduled_for=None):
        raise AssertionError("should not reach SkyPilot with an overlong post")

    monkeypatch.setattr("web.routers.linkedin_to_bluesky.create_post", fake_create_post)

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/send",
        data={"state": state, "post_0_0": "x" * 301},
    )
    assert response.status_code == 200
    assert "301 characters" in response.text
    assert list_content_history(brand.id, "bluesky") == []


def test_mark_used_group_with_overlong_edited_post_shows_friendly_error(client):
    brand = _create_brand()
    results_html = _generate(client, brand.id)

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/mark-used",
        data={"state": state, "post_0_0": "x" * 301},
    )
    assert response.status_code == 200
    assert "301 characters" in response.text
    assert list_content_history(brand.id, "bluesky") == []


def test_wizard_state_round_trips_after_an_edit_exceeds_300_chars(client):
    """Regression test for the actual reported bug: editing a post past 300
    chars used to get silently accepted into the in-memory state (no
    validation on plain attribute assignment) and then re-serialized into
    the next response's hidden `state` field — which parsed fine with
    model_dump_json() but then hard-failed with an unhandled 500 on
    LinkedInToBlueskyWizardState.model_validate_json() the *next* time any
    button on the page was clicked, since that path fully re-validates.
    add_post_to_group neither checks length nor sends anywhere, so it's a
    clean way to bake an overlong edit into the rendered state; a second,
    unrelated request must then still succeed instead of 500ing.
    """
    brand = _create_brand()
    results_html = _generate(client, brand.id)

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    poisoned_response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/add-post",
        data={"state": state, "post_0_0": "x" * 301},
    )
    assert poisoned_response.status_code == 200
    poisoned_state = _extract_state(poisoned_response.text)
    assert json.loads(poisoned_state)["groups"][0]["posts"][0] == "x" * 301

    second_response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/add-post",
        data={"state": poisoned_state, **_pending_post_fields(poisoned_state)},
    )
    assert second_response.status_code == 200


def test_delete_group_removes_it(client):
    brand = _create_brand()
    results_html = _generate(client, brand.id)

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/delete",
        data={"state": state, **_pending_post_fields(state)},
    )
    updated_state = json.loads(_extract_state(response.text))
    assert updated_state["groups"] == []
    assert "Nothing left to review" in response.text


def test_add_post_to_group_extends_it_into_a_thread(client):
    brand = _create_brand()
    results_html = _generate(client, brand.id)

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/add-post",
        data={"state": state, **_pending_post_fields(state)},
    )
    updated_state = json.loads(_extract_state(response.text))
    assert updated_state["groups"][0]["posts"] == ["a", ""]
    assert 'name="post_0_1"' in response.text


def test_delete_post_removes_whole_group_when_it_empties(client):
    brand = _create_brand()
    results_html = _generate(client, brand.id)  # one group, one post under TestModel

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/posts/0/delete",
        data={"state": state, **_pending_post_fields(state)},
    )
    updated_state = json.loads(_extract_state(response.text))
    assert updated_state["groups"] == []


def test_add_group_creates_a_new_empty_pending_group(client):
    brand = _create_brand()
    results_html = _generate(client, brand.id)

    state = _extract_state(results_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/add",
        data={"state": state, **_pending_post_fields(state)},
    )
    updated_state = json.loads(_extract_state(response.text))
    assert len(updated_state["groups"]) == 2
    assert updated_state["groups"][1]["posts"] == [""]
    assert updated_state["groups"][1]["status"] == "pending"


def test_regenerate_keeps_sent_group_untouched_and_replaces_pending(client, monkeypatch):
    brand = _create_brand(skypilot_id="sky-abc")
    results_html = _generate(client, brand.id)

    async def fake_create_post(account_id, texts, scheduled_for=None):
        return {"id": "post-999"}

    monkeypatch.setattr("web.routers.linkedin_to_bluesky.create_post", fake_create_post)

    state = _extract_state(results_html)
    group_id = _group_ids(results_html)[0]
    sent_response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/{group_id}/send",
        data={"state": state, **_pending_post_fields(state)},
    )
    sent_state = json.loads(_extract_state(sent_response.text))
    assert len(sent_state["groups"]) == 1
    assert sent_state["groups"][0]["status"] == "sent"

    # Add a fresh pending group manually, then regenerate — it should be
    # replaced by a freshly-generated pending group, while the already-sent
    # one stays exactly as it is.
    state = _extract_state(sent_response.text)
    add_response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/groups/add", data={"state": state}
    )
    state = _extract_state(add_response.text)
    regen_response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/regenerate",
        data={"state": state, "feedback": "make it punchier"},
    )
    regenerated_html = _poll(client, brand.id, regen_response.text)
    regenerated_state = json.loads(_extract_state(regenerated_html))

    statuses = [g["status"] for g in regenerated_state["groups"]]
    assert statuses.count("sent") == 1
    assert statuses.count("pending") == 1  # the manually-added empty group was discarded
    sent_group = next(g for g in regenerated_state["groups"] if g["status"] == "sent")
    assert sent_group["id"] == group_id
    assert sent_group["posts"] == ["a"]  # untouched


def test_regenerate_feedback_is_cumulative(client, monkeypatch):
    """Every regenerate's feedback persists and all still apply to future
    regenerates — same guarantee as the LinkedIn wizard's feedback_log.
    """
    brand = _create_brand()
    results_html = _generate(client, brand.id)

    captured_deps = []
    import agent.agents.linkedin_to_bluesky as l2b_module

    original = l2b_module.run_linkedin_to_bluesky_agent

    async def spy(deps, on_phase=None):
        captured_deps.append(deps)
        return await original(deps, on_phase=on_phase)

    monkeypatch.setattr("web.routers.linkedin_to_bluesky.run_linkedin_to_bluesky_agent", spy)

    state = _extract_state(results_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/regenerate",
        data={"state": state, "feedback": "make it punchier"},
    )
    first_regen_html = _poll(client, brand.id, response.text)

    state = _extract_state(first_regen_html)
    response = client.post(
        f"/brands/{brand.id}/linkedin-to-bluesky/regenerate",
        data={"state": state, "feedback": "avoid emoji"},
    )
    _poll(client, brand.id, response.text)

    assert len(captured_deps) == 2
    assert captured_deps[1].feedback_log == ["make it punchier", "avoid emoji"]
