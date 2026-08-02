"""Newsletter agent endpoints: generate/regenerate (same handler), mark as
used. "Copy to Clipboard" is pure client-side JS — no route for it.

Generation runs as a background job (see web/jobs.py) rather than being
awaited directly in the POST handler: it's two sequential model calls (body,
then subjects — see agent/agents/newsletter.py) that can take tens of
seconds combined, and the UI polls /status to show which phase is running.
"""

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic_ai import UsageLimitExceeded

from agent.agents import NewsletterDeps, run_newsletter_agent
from agent.logging import get_logger
from db.models import Brand, NewsletterContent, SubjectPair as StoredSubjectPair
from db.repository import (
    add_content_history,
    get_brand,
    get_newsletter_settings,
    recent_newsletter_summaries,
)
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


@router.get("/brands/{brand_id}/newsletter", response_class=HTMLResponse)
async def newsletter_page(request: Request, brand_id: str) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    context = base_context(active_brand_id=brand_id, active_agent="newsletter")
    context.update({"request": request, "brand": brand})
    return templates.TemplateResponse(request, "newsletter.html", context)


async def _run_newsletter_job(job_id: str, brand: Brand, prompt: str) -> None:
    newsletter_settings = get_newsletter_settings(brand.id)
    recent_summaries = recent_newsletter_summaries(brand.id)
    deps = NewsletterDeps(
        brand=brand, settings=newsletter_settings, recent_summaries=recent_summaries
    )

    def on_phase(phase: str) -> None:
        update_job(job_id, phase=phase)

    try:
        output = await run_newsletter_agent(prompt, deps, on_phase=on_phase)
        update_job(job_id, status="done", result=output)
    except UsageLimitExceeded:
        update_job(
            job_id, status="error", error="Generation hit its usage limit — try a shorter prompt."
        )
    except Exception:
        logger.exception("Newsletter generation failed", extra={"brand_id": brand.id})
        update_job(
            job_id,
            status="error",
            error="Something went wrong generating this newsletter — please try again.",
        )


@router.post("/brands/{brand_id}/newsletter/generate", response_class=HTMLResponse)
async def generate(request: Request, brand_id: str, prompt: str = Form(...)) -> HTMLResponse:
    """Also serves as "Regenerate" — the result partial resubmits the same
    original prompt via a hidden field, discarding any edits. Never touches
    history. Kicks off the job and returns the polling partial immediately;
    the actual result is rendered by status() below once the job finishes.
    """
    brand = _get_brand_or_404(brand_id)
    job_id = create_job()
    run_in_background(_run_newsletter_job(job_id, brand, prompt))

    return templates.TemplateResponse(
        request,
        "partials/newsletter_polling.html",
        {
            "brand_id": brand_id,
            "job_id": job_id,
            "original_prompt": prompt,
            "phase": "Generating newsletter body",
        },
    )


@router.get("/brands/{brand_id}/newsletter/status/{job_id}", response_class=HTMLResponse)
async def status(request: Request, brand_id: str, job_id: str, prompt: str = "") -> HTMLResponse:
    job = get_job(job_id)
    if job is None:
        return render_error(request, "Lost track of that generation — please try again.")

    if job.status == "running":
        return templates.TemplateResponse(
            request,
            "partials/newsletter_polling.html",
            {"brand_id": brand_id, "job_id": job_id, "original_prompt": prompt, "phase": job.phase},
        )

    pop_job(job_id)
    if job.status == "error":
        return render_error(request, job.error or "Something went wrong.")

    output = job.result
    return templates.TemplateResponse(
        request,
        "partials/newsletter_result.html",
        {
            "brand_id": brand_id,
            "body_html": output.body_html,
            "subject_pairs": output.subject_pairs,
            "original_prompt": prompt,
        },
    )


@router.post("/brands/{brand_id}/newsletter/mark-used", response_class=HTMLResponse)
async def mark_used(request: Request, brand_id: str) -> HTMLResponse:
    _get_brand_or_404(brand_id)
    form = await request.form()
    body_html = str(form.get("body_html", ""))
    count = int(form.get("subject_count", "0") or "0")
    subject_pairs = [
        StoredSubjectPair(
            subject=str(form.get(f"subject_{i}", "")),
            description=str(form.get(f"description_{i}", "")),
        )
        for i in range(count)
    ]

    if not body_html.strip() or not subject_pairs:
        raise HTTPException(400, "Missing newsletter content")

    entry = add_content_history(
        brand_id, "newsletter", NewsletterContent(body_html=body_html, subject_pairs=subject_pairs)
    )
    return templates.TemplateResponse(request, "partials/newsletter_sent.html", {"entry": entry})
