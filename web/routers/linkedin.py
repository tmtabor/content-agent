"""LinkedIn agent endpoints: a 6-step guided wizard (concept -> hooks ->
hashtags -> outline -> draft -> polish). The whole wizard's state is a
single JSON-serialized hidden form field (`LinkedInWizardState`) round-
tripped through every step — see agent/agents/linkedin.py for the model
and CLAUDE.md for why (a 6-step wizard with lists and a feedback log has
too much shape for several individual hidden fields to stay readable, the
pattern Bluesky/Newsletter use).

Caching/regeneration rule (deliberate simplification — see CLAUDE.md):
every `.../generate` endpoint only calls the model if its target step has
no cached content yet in the submitted state; `/back` never regenerates,
it just re-renders whatever's cached; `.../regenerate` always overwrites
the current step's cache but does not cascade-invalidate later steps.
The one exception is `polish/generate`, which always regenerates — a
stale polish result for text the user has since edited would be actively
misleading, not just out of sync.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic_ai import UsageLimitExceeded

from agent.agents import (
    LinkedInDeps,
    LinkedInWizardState,
    run_linkedin_draft_agent,
    run_linkedin_hashtags_agent,
    run_linkedin_hooks_agent,
    run_linkedin_outline_agent,
    run_linkedin_polish_agent,
)
from agent.logging import get_logger
from db.models import Brand, LinkedInContent
from db.repository import (
    add_content_history,
    get_brand,
    get_linkedin_settings,
    recent_linkedin_texts,
)
from web.context import base_context
from web.jinja import templates
from web.jobs import create_job, get_job, pop_job, run_in_background, update_job
from web.rendering import render_error

router = APIRouter()
logger = get_logger(__name__)

_STEP_ORDER = ["concept", "hooks", "hashtags", "outline", "draft", "polish"]
_STEP_TEMPLATES = {
    "concept": "partials/linkedin_concept.html",
    "hooks": "partials/linkedin_hooks.html",
    "hashtags": "partials/linkedin_hashtags.html",
    "outline": "partials/linkedin_outline.html",
    "draft": "partials/linkedin_draft.html",
    "polish": "partials/linkedin_polish.html",
}


def _get_brand_or_404(brand_id: str) -> Brand:
    brand = get_brand(brand_id)
    if brand is None:
        raise HTTPException(404, "Brand not found")
    return brand


def _build_deps(brand: Brand, state: LinkedInWizardState) -> LinkedInDeps:
    return LinkedInDeps(
        brand=brand,
        settings=get_linkedin_settings(brand.id),
        recent_posts=recent_linkedin_texts(brand.id),
        concept=state.concept,
        accepted_hook=state.accepted_hook,
        accepted_hashtags=state.accepted_hashtags,
        outline_bullets=state.outline_bullets,
        draft_text=state.draft_text,
        feedback_log=list(state.feedback_log),
    )


def _render_step(request: Request, brand_id: str, state: LinkedInWizardState) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        _STEP_TEMPLATES[state.step],
        {
            "request": request,
            "brand_id": brand_id,
            "state": state,
            "state_json": state.model_dump_json(),
        },
    )


def _render_polling(
    request: Request, brand_id: str, job_id: str, phase: str, step: str
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/linkedin_polling.html",
        {"request": request, "brand_id": brand_id, "job_id": job_id, "phase": phase, "step": step},
    )


async def _run_step_job(
    job_id: str,
    brand_id: str,
    deps: LinkedInDeps,
    state: LinkedInWizardState,
    step_label: str,
    mutate: Callable[[LinkedInDeps, LinkedInWizardState], Awaitable[None]],
) -> None:
    """Shared job body for every generate/regenerate endpoint below: run
    `mutate` (which calls the relevant agent and writes its result onto
    `state`), then store the *complete* updated state as the job result —
    unlike newsletter's polling, which round-trips its one string prompt
    through the status URL, LinkedIn's state is too large for that to stay
    clean, so it's carried in this closure instead and handed back whole.
    """
    try:
        await mutate(deps, state)
        update_job(job_id, status="done", result=state)
    except UsageLimitExceeded:
        update_job(
            job_id, status="error", error="Generation hit its usage limit — try a shorter prompt."
        )
    except Exception:
        logger.exception(f"LinkedIn {step_label} generation failed", extra={"brand_id": brand_id})
        update_job(
            job_id,
            status="error",
            error=f"Something went wrong generating {step_label} — please try again.",
        )


async def _mutate_hooks(deps: LinkedInDeps, state: LinkedInWizardState) -> None:
    state.hook_options = await run_linkedin_hooks_agent(deps)


async def _mutate_hashtags(deps: LinkedInDeps, state: LinkedInWizardState) -> None:
    state.hashtag_options = await run_linkedin_hashtags_agent(deps)


async def _mutate_outline(deps: LinkedInDeps, state: LinkedInWizardState) -> None:
    state.outline_bullets = await run_linkedin_outline_agent(deps)


async def _mutate_draft(deps: LinkedInDeps, state: LinkedInWizardState) -> None:
    state.draft_text = await run_linkedin_draft_agent(deps)


async def _mutate_polish(deps: LinkedInDeps, state: LinkedInWizardState) -> None:
    state.polish_original = state.draft_text
    state.polish_result = await run_linkedin_polish_agent(deps)


def _append_feedback(state: LinkedInWizardState, feedback: str) -> None:
    """Every regenerate's instructions persist for the rest of the session
    (per the spec's "retains in its context every accepted step and
    feedback") — not just used for this one call and discarded.
    """
    if feedback.strip():
        state.feedback_log.append(feedback.strip())


# --- Entry point ---


@router.get("/brands/{brand_id}/linkedin", response_class=HTMLResponse)
async def linkedin_page(request: Request, brand_id: str) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    state = LinkedInWizardState()
    context = base_context(active_brand_id=brand_id, active_agent="linkedin")
    context.update(
        {
            "request": request,
            "brand": brand,
            "brand_id": brand_id,
            "state": state,
            "state_json": state.model_dump_json(),
        }
    )
    return templates.TemplateResponse(request, "linkedin.html", context)


# --- Step 1: Hooks ---


@router.post("/brands/{brand_id}/linkedin/hooks/generate", response_class=HTMLResponse)
async def generate_hooks(
    request: Request, brand_id: str, state: str = Form(...), concept: str = Form(...)
) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    wizard_state = LinkedInWizardState.model_validate_json(state)
    wizard_state.concept = concept
    wizard_state.step = "hooks"

    if wizard_state.hook_options:
        return _render_step(request, brand_id, wizard_state)

    deps = _build_deps(brand, wizard_state)
    job_id = create_job()
    update_job(job_id, phase="Generating hooks", step="hooks")
    run_in_background(_run_step_job(job_id, brand_id, deps, wizard_state, "hooks", _mutate_hooks))
    return _render_polling(request, brand_id, job_id, "Generating hooks", "hooks")


@router.post("/brands/{brand_id}/linkedin/hooks/regenerate", response_class=HTMLResponse)
async def regenerate_hooks(
    request: Request,
    brand_id: str,
    state: str = Form(...),
    regen_instructions: str = Form(""),
) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    wizard_state = LinkedInWizardState.model_validate_json(state)
    wizard_state.accepted_hook = ""  # the old selection may not apply to the new set
    _append_feedback(wizard_state, regen_instructions)

    deps = _build_deps(brand, wizard_state)
    deps.previous_options = [hook.hook_text for hook in wizard_state.hook_options]
    deps.regen_instructions = regen_instructions

    job_id = create_job()
    update_job(job_id, phase="Generating hooks", step="hooks")
    run_in_background(_run_step_job(job_id, brand_id, deps, wizard_state, "hooks", _mutate_hooks))
    return _render_polling(request, brand_id, job_id, "Generating hooks", "hooks")


# --- Step 2: Hashtags ---


@router.post("/brands/{brand_id}/linkedin/hashtags/generate", response_class=HTMLResponse)
async def generate_hashtags(request: Request, brand_id: str) -> HTMLResponse:
    """Reads `selected_hook` (radio: "0"-"4" or "custom") plus the matching
    `hook_text_{selected_hook}` field rather than a single typed `accepted_hook`
    param — there's no plain-HTML way to make "whichever textarea the user
    picked via radio" submit under one already-named field without JS, and
    this app prefers server-side field-picking over inline JS for that (see
    web/routers/bluesky.py's post_0..post_N handling for the same pattern).
    """
    brand = _get_brand_or_404(brand_id)
    form = await request.form()
    wizard_state = LinkedInWizardState.model_validate_json(str(form["state"]))
    selected = str(form.get("selected_hook", "custom"))
    wizard_state.accepted_hook = str(form.get(f"hook_text_{selected}", "")).strip()
    wizard_state.step = "hashtags"

    if wizard_state.hashtag_options:
        return _render_step(request, brand_id, wizard_state)

    deps = _build_deps(brand, wizard_state)
    job_id = create_job()
    update_job(job_id, phase="Generating hashtags", step="hashtags")
    run_in_background(
        _run_step_job(job_id, brand_id, deps, wizard_state, "hashtags", _mutate_hashtags)
    )
    return _render_polling(request, brand_id, job_id, "Generating hashtags", "hashtags")


@router.post("/brands/{brand_id}/linkedin/hashtags/regenerate", response_class=HTMLResponse)
async def regenerate_hashtags(
    request: Request,
    brand_id: str,
    state: str = Form(...),
    regen_instructions: str = Form(""),
) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    wizard_state = LinkedInWizardState.model_validate_json(state)
    wizard_state.accepted_hashtags = []
    _append_feedback(wizard_state, regen_instructions)

    deps = _build_deps(brand, wizard_state)
    deps.previous_options = [tag.hashtag for tag in wizard_state.hashtag_options]
    deps.regen_instructions = regen_instructions

    job_id = create_job()
    update_job(job_id, phase="Generating hashtags", step="hashtags")
    run_in_background(
        _run_step_job(job_id, brand_id, deps, wizard_state, "hashtags", _mutate_hashtags)
    )
    return _render_polling(request, brand_id, job_id, "Generating hashtags", "hashtags")


def _parse_bullets(outline_text: str) -> list[str]:
    return [line.strip() for line in outline_text.splitlines() if line.strip()]


# --- Step 3: Outline ---


@router.post("/brands/{brand_id}/linkedin/outline/generate", response_class=HTMLResponse)
async def generate_outline(request: Request, brand_id: str, state: str = Form(...)) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    wizard_state = LinkedInWizardState.model_validate_json(state)

    form = await request.form()
    accepted = [str(form[f"hashtag_{i}"]) for i in range(10) if form.get(f"hashtag_{i}")]
    custom = str(form.get("custom_hashtags", ""))
    accepted += [tag.strip() for tag in custom.replace(",", " ").split() if tag.strip()]
    wizard_state.accepted_hashtags = accepted
    wizard_state.step = "outline"

    if wizard_state.outline_bullets:
        return _render_step(request, brand_id, wizard_state)

    deps = _build_deps(brand, wizard_state)
    job_id = create_job()
    update_job(job_id, phase="Generating outline", step="outline")
    run_in_background(
        _run_step_job(job_id, brand_id, deps, wizard_state, "outline", _mutate_outline)
    )
    return _render_polling(request, brand_id, job_id, "Generating outline", "outline")


@router.post("/brands/{brand_id}/linkedin/outline/regenerate", response_class=HTMLResponse)
async def regenerate_outline(
    request: Request,
    brand_id: str,
    state: str = Form(...),
    outline_text: str = Form(""),
    feedback: str = Form(""),
) -> HTMLResponse:
    """`outline_text` is the outline exactly as currently shown/edited on
    screen — read it (instead of ignoring it and only looking at `state`,
    which holds whatever was last *generated*, not the user's edits) so
    regeneration has something concrete to revise. See
    agent/agents/linkedin.py's outline_context, which uses
    deps.outline_bullets as the "current outline to revise" anchor.
    """
    brand = _get_brand_or_404(brand_id)
    wizard_state = LinkedInWizardState.model_validate_json(state)
    wizard_state.outline_bullets = _parse_bullets(outline_text)
    _append_feedback(wizard_state, feedback)

    deps = _build_deps(brand, wizard_state)
    job_id = create_job()
    update_job(job_id, phase="Generating outline", step="outline")
    run_in_background(
        _run_step_job(job_id, brand_id, deps, wizard_state, "outline", _mutate_outline)
    )
    return _render_polling(request, brand_id, job_id, "Generating outline", "outline")


# --- Step 4: Draft ---


@router.post("/brands/{brand_id}/linkedin/draft/generate", response_class=HTMLResponse)
async def generate_draft(
    request: Request, brand_id: str, state: str = Form(...), outline_text: str = Form(...)
) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    wizard_state = LinkedInWizardState.model_validate_json(state)
    wizard_state.outline_bullets = _parse_bullets(outline_text)
    wizard_state.step = "draft"

    if wizard_state.draft_text:
        return _render_step(request, brand_id, wizard_state)

    deps = _build_deps(brand, wizard_state)
    job_id = create_job()
    update_job(job_id, phase="Generating draft", step="draft")
    run_in_background(_run_step_job(job_id, brand_id, deps, wizard_state, "draft", _mutate_draft))
    return _render_polling(request, brand_id, job_id, "Generating draft", "draft")


@router.post("/brands/{brand_id}/linkedin/draft/regenerate", response_class=HTMLResponse)
async def regenerate_draft(
    request: Request,
    brand_id: str,
    state: str = Form(...),
    draft_text: str = Form(""),
    feedback: str = Form(""),
) -> HTMLResponse:
    """`draft_text` is the draft exactly as currently shown/edited on
    screen — same reasoning as regenerate_outline above. Previously this
    was silently dropped (only `state`/`feedback` were read), so
    regeneration had no idea what it was supposed to be revising and
    effectively rewrote from the outline blind, every time.
    """
    brand = _get_brand_or_404(brand_id)
    wizard_state = LinkedInWizardState.model_validate_json(state)
    wizard_state.draft_text = draft_text
    _append_feedback(wizard_state, feedback)

    deps = _build_deps(brand, wizard_state)
    job_id = create_job()
    update_job(job_id, phase="Generating draft", step="draft")
    run_in_background(_run_step_job(job_id, brand_id, deps, wizard_state, "draft", _mutate_draft))
    return _render_polling(request, brand_id, job_id, "Generating draft", "draft")


# --- Step 5: Polish (final pass — always regenerates, no cache, no regenerate button) ---


@router.post("/brands/{brand_id}/linkedin/polish/generate", response_class=HTMLResponse)
async def generate_polish(
    request: Request, brand_id: str, state: str = Form(...), draft_text: str = Form(...)
) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    wizard_state = LinkedInWizardState.model_validate_json(state)
    wizard_state.draft_text = draft_text
    wizard_state.step = "polish"

    deps = _build_deps(brand, wizard_state)
    job_id = create_job()
    update_job(job_id, phase="Reviewing post", step="polish")
    run_in_background(_run_step_job(job_id, brand_id, deps, wizard_state, "polish", _mutate_polish))
    return _render_polling(request, brand_id, job_id, "Reviewing post", "polish")


# --- Polling ---


@router.get("/brands/{brand_id}/linkedin/status/{job_id}", response_class=HTMLResponse)
async def status(request: Request, brand_id: str, job_id: str) -> HTMLResponse:
    job = get_job(job_id)
    if job is None:
        return render_error(request, "Lost track of that generation — please try again.")

    if job.status == "running":
        return _render_polling(request, brand_id, job_id, job.phase, job.step)

    pop_job(job_id)
    if job.status == "error":
        return render_error(request, job.error or "Something went wrong.")

    return _render_step(request, brand_id, job.result)


# --- Back navigation (always synchronous — content is already cached) ---


@router.post("/brands/{brand_id}/linkedin/back", response_class=HTMLResponse)
async def back(request: Request, brand_id: str, state: str = Form(...)) -> HTMLResponse:
    _get_brand_or_404(brand_id)
    wizard_state = LinkedInWizardState.model_validate_json(state)
    current_index = _STEP_ORDER.index(wizard_state.step)
    if current_index > 0:
        wizard_state.step = _STEP_ORDER[current_index - 1]
    return _render_step(request, brand_id, wizard_state)


# --- Mark as used (from either polish panel) ---


@router.post("/brands/{brand_id}/linkedin/mark-used", response_class=HTMLResponse)
async def mark_used(request: Request, brand_id: str, post_text: str = Form(...)) -> HTMLResponse:
    _get_brand_or_404(brand_id)
    if not post_text.strip():
        raise HTTPException(400, "Missing post text")

    entry = add_content_history(brand_id, "linkedin", LinkedInContent(post_text=post_text))
    return templates.TemplateResponse(request, "partials/linkedin_saved.html", {"entry": entry})
