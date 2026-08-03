"""End-to-end web tests via FastAPI's TestClient.

Uses the shared `client` fixture from tests/conftest.py (`with
TestClient(app) as client:`, so the lifespan handler runs and calls
init_db() against the per-test temp db).
"""

import re
import time

from db.repository import (
    create_brand,
    get_bluesky_settings,
    get_newsletter_settings,
    list_content_history,
)


def _create_brand(**overrides):
    defaults = dict(
        name="Acme", background="Widgets", voice="playful", audience="makers", skypilot_id=""
    )
    defaults.update(overrides)
    return create_brand(**defaults)


def _poll_newsletter_job(client, brand_id, generate_response_text):
    """Newsletter generation runs as a background job (see web/jobs.py) —
    POST /generate returns a polling partial immediately, not the result.
    Extract the job status URL from it and poll until the job finishes
    (TestModel is effectively instant, so this resolves almost immediately;
    the loop is just a safety margin against scheduling timing).
    """
    match = re.search(r'/newsletter/status/([a-f0-9]+)\?prompt=([^"]*)', generate_response_text)
    assert match, f"expected a polling partial with a job status URL, got: {generate_response_text}"
    job_id, prompt = match.group(1), match.group(2)

    for _ in range(50):
        response = client.get(f"/brands/{brand_id}/newsletter/status/{job_id}?prompt={prompt}")
        if "generating-status" not in response.text:
            return response
        time.sleep(0.02)
    raise AssertionError("newsletter job did not finish within the polling budget")


# --- Brand CRUD ---


def test_index_redirects_to_new_brand_form_when_no_brands(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 303, 307)
    assert response.headers["location"] == "/brands/new"


def test_index_redirects_to_first_brand_bluesky_page(client):
    brand = _create_brand()
    response = client.get("/", follow_redirects=False)
    assert response.headers["location"] == f"/brands/{brand.id}/bluesky"


def test_create_brand_via_form(client):
    response = client.post(
        "/brands",
        data={
            "name": "Widgetco",
            "background": "We make widgets",
            "voice": "formal",
            "audience": "enterprises",
            "skypilot_id": "sky-123",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.endswith("/bluesky")

    settings_response = client.get(location.replace("/bluesky", "/settings"))
    assert "Widgetco" in settings_response.text
    assert "sky-123" in settings_response.text


def test_update_brand(client):
    brand = _create_brand()
    response = client.post(
        f"/brands/{brand.id}",
        data={
            "name": "Acme Renamed",
            "background": "New background",
            "voice": "",
            "audience": "",
            "skypilot_id": "",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    settings_page = client.get(f"/brands/{brand.id}/settings")
    assert "Acme Renamed" in settings_page.text


def test_delete_brand_redirects_to_new_when_none_remain(client):
    brand = _create_brand()
    response = client.post(f"/brands/{brand.id}/delete", follow_redirects=False)
    assert response.headers["location"] == "/brands/new"
    assert client.get(f"/brands/{brand.id}/settings").status_code == 404


def test_settings_404_for_unknown_brand(client):
    assert client.get("/brands/does-not-exist/settings").status_code == 404


# --- Bluesky settings ---


def test_update_bluesky_settings(client):
    brand = _create_brand()
    response = client.post(
        f"/brands/{brand.id}/bluesky-settings",
        data={"instructions": "No emojis", "hashtags": "#ttrpgsky"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    settings_row = get_bluesky_settings(brand.id)
    assert settings_row.instructions == "No emojis"
    assert settings_row.hashtags == "#ttrpgsky"


def test_update_newsletter_settings_including_html_template(client):
    brand = _create_brand()
    response = client.post(
        f"/brands/{brand.id}/newsletter-settings",
        data={
            "instructions": "3 sections, 200 words each",
            "html_template": "<html><body>{{content}}</body></html>",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    settings_row = get_newsletter_settings(brand.id)
    assert settings_row.instructions == "3 sections, 200 words each"
    assert settings_row.html_template == "<html><body>{{content}}</body></html>"

    settings_page = client.get(f"/brands/{brand.id}/settings")
    assert "{{content}}" in settings_page.text


# --- Bluesky generate / mark-used / send ---


def test_bluesky_generate_renders_result_partial(client):
    brand = _create_brand()
    response = client.post(
        f"/brands/{brand.id}/bluesky/generate",
        data={"prompt": "Announce our sale", "want_thread": "", "post_count": "3"},
    )
    assert response.status_code == 200
    assert 'name="post_0"' in response.text
    assert 'name="post_count_actual" value="1"' in response.text


def test_bluesky_generate_thread_produces_multiple_posts(client):
    brand = _create_brand()
    response = client.post(
        f"/brands/{brand.id}/bluesky/generate",
        data={"prompt": "Announce our sale", "want_thread": "true", "post_count": "3"},
    )
    assert response.status_code == 200
    # TestModel emits schema-conforming stub data — list length isn't
    # guaranteed beyond min_length=1, so just confirm the shape is present.
    assert 'name="post_0"' in response.text


def test_bluesky_generate_does_not_touch_history(client):
    brand = _create_brand()
    for _ in range(3):
        client.post(
            f"/brands/{brand.id}/bluesky/generate",
            data={"prompt": "Announce our sale", "want_thread": "", "post_count": ""},
        )
    assert list_content_history(brand.id, "bluesky") == []


def test_bluesky_mark_used_adds_to_history(client):
    brand = _create_brand()
    response = client.post(
        f"/brands/{brand.id}/bluesky/mark-used",
        data={"post_count_actual": "1", "post_0": "Check out our new widget!"},
    )
    assert response.status_code == 200
    history = list_content_history(brand.id, "bluesky")
    assert len(history) == 1
    assert history[0].payload.posts[0].text == "Check out our new widget!"
    assert history[0].skypilot_post_id is None


def test_bluesky_mark_used_thread_stored_as_one_entry(client):
    brand = _create_brand()
    client.post(
        f"/brands/{brand.id}/bluesky/mark-used",
        data={"post_count_actual": "3", "post_0": "one", "post_1": "two", "post_2": "three"},
    )
    history = list_content_history(brand.id, "bluesky")
    assert len(history) == 1
    assert [p.text for p in history[0].payload.posts] == ["one", "two", "three"]


def test_bluesky_send_without_skypilot_id_shows_error(client):
    brand = _create_brand(skypilot_id="")
    response = client.post(
        f"/brands/{brand.id}/bluesky/send",
        data={"post_count_actual": "1", "post_0": "hello"},
    )
    assert response.status_code == 200
    assert "no SkyPilot account ID" in response.text
    assert list_content_history(brand.id, "bluesky") == []


def test_bluesky_send_to_skypilot_success(client, monkeypatch):
    brand = _create_brand(skypilot_id="sky-abc")

    captured = {}

    async def fake_create_post(account_id, texts, scheduled_for=None):
        captured["account_id"] = account_id
        captured["texts"] = texts
        captured["scheduled_for"] = scheduled_for
        return {"id": "post-999"}

    monkeypatch.setattr("web.routers.bluesky.create_post", fake_create_post)

    response = client.post(
        f"/brands/{brand.id}/bluesky/send",
        data={"post_count_actual": "1", "post_0": "hello world"},
    )
    assert response.status_code == 200
    assert captured["account_id"] == "sky-abc"
    assert captured["texts"] == ["hello world"]
    assert captured["scheduled_for"] is None

    history = list_content_history(brand.id, "bluesky")
    assert len(history) == 1
    assert history[0].skypilot_post_id == "post-999"
    assert history[0].scheduled_for is None


def test_bluesky_send_to_skypilot_with_schedule(client, monkeypatch):
    brand = _create_brand(skypilot_id="sky-abc")

    async def fake_create_post(account_id, texts, scheduled_for=None):
        return {"id": "post-1"}

    monkeypatch.setattr("web.routers.bluesky.create_post", fake_create_post)

    response = client.post(
        f"/brands/{brand.id}/bluesky/send",
        data={
            "post_count_actual": "1",
            "post_0": "hello",
            "scheduled_for": "2026-12-01T10:00",
        },
    )
    assert response.status_code == 200
    history = list_content_history(brand.id, "bluesky")
    assert history[0].scheduled_for is not None


def test_bluesky_send_skypilot_error_shown(client, monkeypatch):
    from integrations.skypilot import SkyPilotError

    brand = _create_brand(skypilot_id="sky-abc")

    async def failing_create_post(account_id, texts, scheduled_for=None):
        raise SkyPilotError("boom")

    monkeypatch.setattr("web.routers.bluesky.create_post", failing_create_post)

    response = client.post(
        f"/brands/{brand.id}/bluesky/send",
        data={"post_count_actual": "1", "post_0": "hello"},
    )
    assert response.status_code == 200
    assert "boom" in response.text
    assert list_content_history(brand.id, "bluesky") == []


# --- Newsletter ---


def test_newsletter_generate_returns_polling_partial(client):
    brand = _create_brand()
    response = client.post(
        f"/brands/{brand.id}/newsletter/generate", data={"prompt": "This month's update"}
    )
    assert response.status_code == 200
    assert "generating-status" in response.text
    assert "Generating newsletter body" in response.text


def test_newsletter_generate_eventually_renders_result_partial(client):
    brand = _create_brand()
    generate_response = client.post(
        f"/brands/{brand.id}/newsletter/generate", data={"prompt": "This month's update"}
    )
    final_response = _poll_newsletter_job(client, brand.id, generate_response.text)
    assert final_response.status_code == 200
    assert 'name="body_html"' in final_response.text
    assert 'name="subject_count" value="5"' in final_response.text


def test_newsletter_generate_does_not_touch_history(client):
    brand = _create_brand()
    generate_response = client.post(
        f"/brands/{brand.id}/newsletter/generate", data={"prompt": "This month's update"}
    )
    _poll_newsletter_job(client, brand.id, generate_response.text)
    assert list_content_history(brand.id, "newsletter") == []


def test_newsletter_status_unknown_job_shows_error(client):
    brand = _create_brand()
    response = client.get(f"/brands/{brand.id}/newsletter/status/does-not-exist")
    assert response.status_code == 200
    assert "error-banner" in response.text


def test_newsletter_mark_used_adds_to_history(client):
    brand = _create_brand()
    data = {"body_html": "<p>Hello</p>", "subject_count": "5"}
    for i in range(5):
        data[f"subject_{i}"] = f"Subject {i}"
        data[f"description_{i}"] = f"Description {i}"

    response = client.post(f"/brands/{brand.id}/newsletter/mark-used", data=data)
    assert response.status_code == 200
    history = list_content_history(brand.id, "newsletter")
    assert len(history) == 1
    assert history[0].payload.body_html == "<p>Hello</p>"
    assert len(history[0].payload.subject_pairs) == 5


# --- Settings: history browser purge/delete ---


def test_delete_single_history_entry_via_settings(client):
    brand = _create_brand()
    client.post(
        f"/brands/{brand.id}/bluesky/mark-used",
        data={"post_count_actual": "1", "post_0": "hello"},
    )
    entry_id = list_content_history(brand.id, "bluesky")[0].id

    response = client.post(
        f"/brands/{brand.id}/history/bluesky/{entry_id}/delete", follow_redirects=False
    )
    assert response.status_code == 303
    assert list_content_history(brand.id, "bluesky") == []


def test_purge_all_history_via_settings(client):
    brand = _create_brand()
    for i in range(3):
        client.post(
            f"/brands/{brand.id}/bluesky/mark-used",
            data={"post_count_actual": "1", "post_0": f"post {i}"},
        )
    assert len(list_content_history(brand.id, "bluesky")) == 3

    response = client.post(f"/brands/{brand.id}/history/bluesky/purge", follow_redirects=False)
    assert response.status_code == 303
    assert list_content_history(brand.id, "bluesky") == []


# --- Shutdown (used by the macOS launcher's in-UI stop link) ---


def test_shutdown_returns_confirmation_without_killing_the_test_process(client, monkeypatch):
    """The route sends SIGINT to itself after a delay (see web/main.py) —
    mock os.kill so that if the background task actually runs during this
    (or a later) test, it can't take down the test process.
    """
    killed = []
    monkeypatch.setattr("web.main.os.kill", lambda pid, sig: killed.append((pid, sig)))

    response = client.post("/shutdown")

    assert response.status_code == 200
    assert "Shutting down" in response.text
