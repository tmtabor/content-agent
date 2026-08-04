"""LinkedIn-to-Bluesky agent endpoints: a 2-state flow (input -> results)
rather than LinkedIn's 6-step wizard — this agent's job is "generate a
heterogeneous batch of posts/threads, then let the user edit/send/delete/add
freely," not a linear sequence of steps. Generation (`/generate`,
`/regenerate`) runs as a background job with polling, same reasoning as
newsletter.py's router: two sequential real model calls (see
agent/agents/linkedin_to_bluesky.py) take real time against the default
local model. Every per-group mutation (send, mark-used, delete, add-post,
delete-post, add-group) never touches the model, so those stay synchronous.

Caching/regeneration rule: Regenerate discards every `pending` group and
replaces them with a fresh generation, but leaves `sent`/`used` groups
untouched — they're already persisted to content_history, and stay visible
in the results view with a read-only badge (see
partials/linkedin_to_bluesky_results.html). Feedback is cumulative across
regenerates (`state.feedback_log`), same as agent/agents/linkedin.py's.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic_ai import UsageLimitExceeded

from agent.agents import (
    LinkedInToBlueskyDeps,
    LinkedInToBlueskyGroup,
    LinkedInToBlueskyWizardState,
    run_linkedin_to_bluesky_agent,
)
from agent.logging import get_logger
from db.models import BlueskyContent, BlueskyPost as StoredBlueskyPost, Brand, LinkedInContent
from db.repository import (
    add_content_history,
    get_bluesky_settings,
    get_brand,
    recent_bluesky_texts,
)
from integrations.skypilot import SkyPilotError, create_post
from web.context import base_context
from web.jinja import templates
from web.jobs import create_job, get_job, pop_job, run_in_background, update_job
from web.rendering import render_error

router = APIRouter()
logger = get_logger(__name__)


def _get_brand_or_404(brand_id: str) -> Brand:
    brand = get_brand(brand_id)
    if brand is None:
        raise HTTPException(404, "Brand not found")
    return brand


def _get_group(state: LinkedInToBlueskyWizardState, group_id: str) -> LinkedInToBlueskyGroup | None:
    for group in state.groups:
        if group.id == group_id:
            return group
    return None


def _sync_pending_edits(form, state: LinkedInToBlueskyWizardState) -> None:
    """Overwrite every pending group's posts with whatever's currently in
    its post_{gi}_{pi} fields, before applying whichever specific mutation
    this request is for — so an edit made to group A isn't lost when the
    user clicks Send on group B. Sent/used groups render read-only (no
    input fields), so there's nothing to sync for them.
    """
    for gi, group in enumerate(state.groups):
        if group.status != "pending":
            continue
        group.posts = [form.get(f"post_{gi}_{pi}", "").strip() for pi in range(len(group.posts))]


def _append_feedback(state: LinkedInToBlueskyWizardState, feedback: str) -> None:
    """Every regenerate's instructions persist for the rest of the session —
    same reasoning as agent/agents/linkedin.py's identical helper.
    """
    if feedback.strip():
        state.feedback_log.append(feedback.strip())


def _render_page(
    request: Request, brand: Brand, brand_id: str, state: LinkedInToBlueskyWizardState
) -> HTMLResponse:
    context = base_context(active_brand_id=brand_id, active_agent="linkedin-to-bluesky")
    context.update(
        {
            "request": request,
            "brand": brand,
            "brand_id": brand_id,
            "state": state,
            "state_json": state.model_dump_json(),
        }
    )
    return templates.TemplateResponse(request, "linkedin_to_bluesky.html", context)


def _render_results(
    request: Request, brand_id: str, state: LinkedInToBlueskyWizardState
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/linkedin_to_bluesky_results.html",
        {
            "request": request,
            "brand_id": brand_id,
            "state": state,
            "state_json": state.model_dump_json(),
        },
    )


def _render_polling(request: Request, brand_id: str, job_id: str, phase: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/linkedin_to_bluesky_polling.html",
        {"request": request, "brand_id": brand_id, "job_id": job_id, "phase": phase},
    )


async def _run_generation_job(
    job_id: str,
    brand_id: str,
    deps: LinkedInToBlueskyDeps,
    state: LinkedInToBlueskyWizardState,
    retained_groups: list[LinkedInToBlueskyGroup],
) -> None:
    """Shared job body for both /generate (retained_groups=[]) and
    /regenerate (retained_groups=whatever was already sent/used) — runs the
    two-call agent, then replaces state.groups with retained + freshly
    generated groups, in that order.
    """
    try:
        new_groups = await run_linkedin_to_bluesky_agent(
            deps, on_phase=lambda phase: update_job(job_id, phase=phase)
        )
        state.groups = [*retained_groups, *new_groups]
        state.step = "results"
        update_job(job_id, status="done", result=state)
    except UsageLimitExceeded:
        update_job(
            job_id, status="error", error="Generation hit its usage limit — try shorter input."
        )
    except Exception:
        logger.exception("LinkedIn-to-Bluesky generation failed", extra={"brand_id": brand_id})
        update_job(
            job_id,
            status="error",
            error="Something went wrong generating these posts — please try again.",
        )


# --- Entry points ---


@router.get("/brands/{brand_id}/linkedin-to-bluesky", response_class=HTMLResponse)
async def linkedin_to_bluesky_page(request: Request, brand_id: str) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    return _render_page(request, brand, brand_id, LinkedInToBlueskyWizardState())


@router.post("/brands/{brand_id}/linkedin-to-bluesky/from-linkedin", response_class=HTMLResponse)
async def from_linkedin(
    request: Request, brand_id: str, source_text: str = Form(...)
) -> HTMLResponse:
    """Receives the handoff from the LinkedIn wizard's Polish step (see
    partials/linkedin_polish.html) — a real full-page navigation, not an
    htmx swap, since it crosses between two top-level agent pages.

    This is the only place a post reaches LinkedIn history for a user who
    adapts it to Bluesky right away without first clicking "Mark as Used"
    on the LinkedIn page itself (see linkedin.py's mark_used) — skipping it
    here silently dropped the LinkedIn post from history entirely, even
    though the user clearly intended to use it (enough to hand it off).
    """
    brand = _get_brand_or_404(brand_id)
    if source_text.strip():
        add_content_history(brand_id, "linkedin", LinkedInContent(post_text=source_text))
    state = LinkedInToBlueskyWizardState(source_text=source_text)
    return _render_page(request, brand, brand_id, state)


# --- Generation ---


@router.post("/brands/{brand_id}/linkedin-to-bluesky/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    brand_id: str,
    source_text: str = Form(...),
    target_count: int | None = Form(None),
    thread_preference: str = Form("no_preference"),
) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    bluesky_settings = get_bluesky_settings(brand_id)
    recent_posts = recent_bluesky_texts(brand_id)

    state = LinkedInToBlueskyWizardState(
        source_text=source_text, target_count=target_count, thread_preference=thread_preference
    )
    deps = LinkedInToBlueskyDeps(
        brand=brand,
        settings=bluesky_settings,
        source_text=source_text,
        target_count=target_count,
        thread_preference=thread_preference,
        recent_posts=recent_posts,
    )

    job_id = create_job()
    update_job(job_id, phase="Analyzing post and planning breakdown")
    run_in_background(_run_generation_job(job_id, brand_id, deps, state, retained_groups=[]))
    return _render_polling(request, brand_id, job_id, "Analyzing post and planning breakdown")


@router.post("/brands/{brand_id}/linkedin-to-bluesky/regenerate", response_class=HTMLResponse)
async def regenerate(
    request: Request,
    brand_id: str,
    state: str = Form(...),
    feedback: str = Form(""),
) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    wizard_state = LinkedInToBlueskyWizardState.model_validate_json(state)
    _append_feedback(wizard_state, feedback)
    retained = [group for group in wizard_state.groups if group.status != "pending"]

    deps = LinkedInToBlueskyDeps(
        brand=brand,
        settings=get_bluesky_settings(brand_id),
        source_text=wizard_state.source_text,
        target_count=wizard_state.target_count,
        thread_preference=wizard_state.thread_preference,
        recent_posts=recent_bluesky_texts(brand_id),
        feedback_log=list(wizard_state.feedback_log),
    )

    job_id = create_job()
    update_job(job_id, phase="Analyzing post and planning breakdown")
    run_in_background(
        _run_generation_job(job_id, brand_id, deps, wizard_state, retained_groups=retained)
    )
    return _render_polling(request, brand_id, job_id, "Analyzing post and planning breakdown")


@router.get("/brands/{brand_id}/linkedin-to-bluesky/status/{job_id}", response_class=HTMLResponse)
async def status(request: Request, brand_id: str, job_id: str) -> HTMLResponse:
    job = get_job(job_id)
    if job is None:
        return render_error(request, "Lost track of that generation — please try again.")

    if job.status == "running":
        return _render_polling(request, brand_id, job_id, job.phase)

    pop_job(job_id)
    if job.status == "error":
        return render_error(request, job.error or "Something went wrong.")

    return _render_results(request, brand_id, job.result)


# --- Per-group mutations (all synchronous — no model call) ---


@router.post(
    "/brands/{brand_id}/linkedin-to-bluesky/groups/{group_id}/send", response_class=HTMLResponse
)
async def send_group(request: Request, brand_id: str, group_id: str) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    form = await request.form()
    wizard_state = LinkedInToBlueskyWizardState.model_validate_json(str(form["state"]))
    _sync_pending_edits(form, wizard_state)

    group = _get_group(wizard_state, group_id)
    if group is None or group.status != "pending":
        return _render_results(request, brand_id, wizard_state)

    texts = [text for text in group.posts if text]
    if not texts:
        return render_error(request, "No post text provided")
    too_long = [text for text in texts if len(text) > 300]
    if too_long:
        return render_error(
            request,
            f"Post is {len(too_long[0])} characters — Bluesky posts must be 300 or fewer. "
            "Edit it and try again.",
        )

    if not brand.skypilot_id:
        return render_error(
            request, "This brand has no SkyPilot account ID configured — set one in Settings first."
        )

    # Keyed by group_id, not a bare "scheduled_for" — every group's Send
    # button lives in the same <form>, so a shared field name would let one
    # group's schedule leak into another's request.
    schedule_raw = str(form.get(f"scheduled_for_{group_id}", "")).strip()
    scheduled_for = datetime.fromisoformat(schedule_raw) if schedule_raw else None

    try:
        response = await create_post(brand.skypilot_id, texts, scheduled_for)
    except SkyPilotError as e:
        return render_error(request, str(e))

    add_content_history(
        brand_id,
        "bluesky",
        BlueskyContent(posts=[StoredBlueskyPost(text=text) for text in texts]),
        skypilot_post_id=str(response.get("id", "")),
        scheduled_for=scheduled_for,
    )
    group.posts = texts
    group.status = "sent"
    group.skypilot_post_id = str(response.get("id", ""))
    group.scheduled_for = scheduled_for
    return _render_results(request, brand_id, wizard_state)


@router.post(
    "/brands/{brand_id}/linkedin-to-bluesky/groups/{group_id}/mark-used",
    response_class=HTMLResponse,
)
async def mark_used_group(request: Request, brand_id: str, group_id: str) -> HTMLResponse:
    _get_brand_or_404(brand_id)
    form = await request.form()
    wizard_state = LinkedInToBlueskyWizardState.model_validate_json(str(form["state"]))
    _sync_pending_edits(form, wizard_state)

    group = _get_group(wizard_state, group_id)
    if group is None or group.status != "pending":
        return _render_results(request, brand_id, wizard_state)

    texts = [text for text in group.posts if text]
    if not texts:
        return render_error(request, "No post text provided")
    too_long = [text for text in texts if len(text) > 300]
    if too_long:
        return render_error(
            request,
            f"Post is {len(too_long[0])} characters — Bluesky posts must be 300 or fewer. "
            "Edit it and try again.",
        )

    add_content_history(
        brand_id, "bluesky", BlueskyContent(posts=[StoredBlueskyPost(text=text) for text in texts])
    )
    group.posts = texts
    group.status = "used"
    return _render_results(request, brand_id, wizard_state)


@router.post(
    "/brands/{brand_id}/linkedin-to-bluesky/groups/{group_id}/delete", response_class=HTMLResponse
)
async def delete_group(request: Request, brand_id: str, group_id: str) -> HTMLResponse:
    _get_brand_or_404(brand_id)
    form = await request.form()
    wizard_state = LinkedInToBlueskyWizardState.model_validate_json(str(form["state"]))
    _sync_pending_edits(form, wizard_state)

    wizard_state.groups = [
        group
        for group in wizard_state.groups
        if not (group.id == group_id and group.status == "pending")
    ]
    return _render_results(request, brand_id, wizard_state)


@router.post(
    "/brands/{brand_id}/linkedin-to-bluesky/groups/{group_id}/add-post",
    response_class=HTMLResponse,
)
async def add_post_to_group(request: Request, brand_id: str, group_id: str) -> HTMLResponse:
    """Appends an empty post to an existing pending group — the mechanism
    for turning a single top-level post into a thread, or extending one.
    """
    _get_brand_or_404(brand_id)
    form = await request.form()
    wizard_state = LinkedInToBlueskyWizardState.model_validate_json(str(form["state"]))
    _sync_pending_edits(form, wizard_state)

    group = _get_group(wizard_state, group_id)
    if group is not None and group.status == "pending":
        group.posts.append("")
    return _render_results(request, brand_id, wizard_state)


@router.post(
    "/brands/{brand_id}/linkedin-to-bluesky/groups/{group_id}/posts/{post_index}/delete",
    response_class=HTMLResponse,
)
async def delete_post_in_group(
    request: Request, brand_id: str, group_id: str, post_index: int
) -> HTMLResponse:
    """Removes one post from a pending group; a group emptied down to zero
    posts is removed entirely. A thread reduced to one remaining post is
    left as-is — it's now just an independent post, same underlying shape.
    """
    _get_brand_or_404(brand_id)
    form = await request.form()
    wizard_state = LinkedInToBlueskyWizardState.model_validate_json(str(form["state"]))
    _sync_pending_edits(form, wizard_state)

    group = _get_group(wizard_state, group_id)
    if group is not None and group.status == "pending" and 0 <= post_index < len(group.posts):
        del group.posts[post_index]
        if not group.posts:
            wizard_state.groups = [g for g in wizard_state.groups if g.id != group_id]
    return _render_results(request, brand_id, wizard_state)


@router.post("/brands/{brand_id}/linkedin-to-bluesky/groups/add", response_class=HTMLResponse)
async def add_group(request: Request, brand_id: str) -> HTMLResponse:
    """Appends a brand-new pending group — one empty independent post for
    the user to type into directly, no model call involved.
    """
    _get_brand_or_404(brand_id)
    form = await request.form()
    wizard_state = LinkedInToBlueskyWizardState.model_validate_json(str(form["state"]))
    _sync_pending_edits(form, wizard_state)

    wizard_state.groups.append(LinkedInToBlueskyGroup(posts=[""]))
    return _render_results(request, brand_id, wizard_state)
