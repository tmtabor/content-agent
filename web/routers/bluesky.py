"""Bluesky agent endpoints: generate/regenerate (same handler — see the
partial's hidden fields), send to SkyPilot, mark as used.
"""

from datetime import datetime

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic_ai import UsageLimitExceeded

from agent.agents import BlueskyDeps, run_bluesky_agent
from agent.logging import get_logger
from db.models import BlueskyContent, BlueskyPost as StoredBlueskyPost, Brand
from db.repository import (
    add_content_history,
    get_bluesky_settings,
    get_brand,
    recent_bluesky_texts,
)
from integrations.skypilot import SkyPilotError, create_post
from web.context import base_context
from web.jinja import templates
from web.rendering import render_error

router = APIRouter()
logger = get_logger(__name__)


def _get_brand_or_404(brand_id: str) -> Brand:
    brand = get_brand(brand_id)
    if brand is None:
        raise HTTPException(404, "Brand not found")
    return brand


def _extract_post_texts(form) -> list[str]:
    """Read the dynamic post_0..post_{n-1} fields the result partial renders
    (count varies with thread length, so these aren't typed Form(...) params).
    """
    count = int(form.get("post_count_actual", "0") or "0")
    texts = [form.get(f"post_{i}", "").strip() for i in range(count)]
    texts = [text for text in texts if text]
    if not texts:
        raise HTTPException(400, "No post text provided")
    return texts


@router.get("/brands/{brand_id}/bluesky", response_class=HTMLResponse)
async def bluesky_page(request: Request, brand_id: str) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    context = base_context(active_brand_id=brand_id, active_agent="bluesky")
    context.update({"request": request, "brand": brand})
    return templates.TemplateResponse(request, "bluesky.html", context)


@router.post("/brands/{brand_id}/bluesky/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    brand_id: str,
    prompt: str = Form(...),
    want_thread: bool = Form(False),
    post_count: int | None = Form(None),
) -> HTMLResponse:
    """Also serves as "Regenerate" — the result partial resubmits these same
    three fields from hidden inputs holding the original generation inputs,
    discarding any edits made to the draft. Neither path touches history.
    """
    brand = _get_brand_or_404(brand_id)
    bluesky_settings = get_bluesky_settings(brand_id)
    recent_posts = recent_bluesky_texts(brand_id)

    deps = BlueskyDeps(
        brand=brand,
        settings=bluesky_settings,
        recent_posts=recent_posts,
        want_thread=want_thread,
        post_count=post_count if want_thread else None,
    )
    try:
        output = await run_bluesky_agent(prompt, deps)
    except UsageLimitExceeded:
        return render_error(request, "Generation hit its usage limit — try a shorter prompt.")
    except Exception:
        logger.exception("Bluesky generation failed", extra={"brand_id": brand_id})
        return render_error(
            request, "Something went wrong generating this post — please try again."
        )

    return templates.TemplateResponse(
        request,
        "partials/bluesky_result.html",
        {
            "brand_id": brand_id,
            "posts": output.posts,
            "original_prompt": prompt,
            "want_thread": want_thread,
            "post_count": post_count,
        },
    )


@router.post("/brands/{brand_id}/bluesky/send", response_class=HTMLResponse)
async def send_to_skypilot(request: Request, brand_id: str) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    form = await request.form()
    texts = _extract_post_texts(form)

    schedule_raw = str(form.get("scheduled_for", "")).strip()
    scheduled_for = datetime.fromisoformat(schedule_raw) if schedule_raw else None

    if not brand.skypilot_id:
        return render_error(
            request, "This brand has no SkyPilot account ID configured — set one in Settings first."
        )

    try:
        response = await create_post(brand.skypilot_id, texts, scheduled_for)
    except SkyPilotError as e:
        return render_error(request, str(e))

    entry = add_content_history(
        brand_id,
        "bluesky",
        BlueskyContent(posts=[StoredBlueskyPost(text=text) for text in texts]),
        skypilot_post_id=str(response.get("id", "")),
        scheduled_for=scheduled_for,
    )
    return templates.TemplateResponse(
        request,
        "partials/bluesky_sent.html",
        {"entry": entry, "scheduled": scheduled_for is not None, "marked_used": False},
    )


@router.post("/brands/{brand_id}/bluesky/mark-used", response_class=HTMLResponse)
async def mark_used(request: Request, brand_id: str) -> HTMLResponse:
    _get_brand_or_404(brand_id)
    form = await request.form()
    texts = _extract_post_texts(form)

    entry = add_content_history(
        brand_id, "bluesky", BlueskyContent(posts=[StoredBlueskyPost(text=text) for text in texts])
    )
    return templates.TemplateResponse(
        request,
        "partials/bluesky_sent.html",
        {"entry": entry, "scheduled": False, "marked_used": True},
    )
