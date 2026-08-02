"""Brand settings: editing brand identity, per-agent settings, Delete Brand,
and the content-history browser (view + purge individual/all entries).
"""

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from db.models import Brand, ContentType
from db.repository import (
    delete_brand,
    delete_content_history_entry,
    get_bluesky_settings,
    get_brand,
    get_newsletter_settings,
    list_brands,
    list_content_history,
    purge_content_history,
    update_bluesky_settings,
    update_brand,
    update_newsletter_settings,
)
from web.context import base_context
from web.jinja import templates

router = APIRouter()


def _get_brand_or_404(brand_id: str) -> Brand:
    brand = get_brand(brand_id)
    if brand is None:
        raise HTTPException(404, "Brand not found")
    return brand


@router.get("/brands/{brand_id}/settings", response_class=HTMLResponse)
async def settings_page(request: Request, brand_id: str) -> HTMLResponse:
    brand = _get_brand_or_404(brand_id)
    context = base_context(active_brand_id=brand_id, active_agent="settings")
    context.update(
        {
            "request": request,
            "brand": brand,
            "bluesky_settings": get_bluesky_settings(brand_id),
            "newsletter_settings": get_newsletter_settings(brand_id),
            "bluesky_history": list_content_history(brand_id, "bluesky"),
            "newsletter_history": list_content_history(brand_id, "newsletter"),
        }
    )
    return templates.TemplateResponse(request, "settings.html", context)


@router.post("/brands/{brand_id}")
async def update_brand_endpoint(
    brand_id: str,
    name: str = Form(...),
    background: str = Form(""),
    voice: str = Form(""),
    audience: str = Form(""),
    skypilot_id: str = Form(""),
) -> RedirectResponse:
    _get_brand_or_404(brand_id)
    update_brand(
        brand_id,
        name=name,
        background=background,
        voice=voice,
        audience=audience,
        skypilot_id=skypilot_id,
    )
    return RedirectResponse(f"/brands/{brand_id}/settings", status_code=303)


@router.post("/brands/{brand_id}/delete")
async def delete_brand_endpoint(brand_id: str) -> RedirectResponse:
    _get_brand_or_404(brand_id)
    delete_brand(brand_id)
    remaining = list_brands()
    if remaining:
        return RedirectResponse(f"/brands/{remaining[0].id}/bluesky", status_code=303)
    return RedirectResponse("/brands/new", status_code=303)


@router.post("/brands/{brand_id}/bluesky-settings")
async def update_bluesky_settings_endpoint(
    brand_id: str,
    instructions: str = Form(""),
    hashtags: str = Form(""),
) -> RedirectResponse:
    _get_brand_or_404(brand_id)
    update_bluesky_settings(brand_id, instructions=instructions, hashtags=hashtags)
    return RedirectResponse(f"/brands/{brand_id}/settings", status_code=303)


@router.post("/brands/{brand_id}/newsletter-settings")
async def update_newsletter_settings_endpoint(
    brand_id: str,
    instructions: str = Form(""),
    html_template: str = Form(""),
) -> RedirectResponse:
    _get_brand_or_404(brand_id)
    update_newsletter_settings(brand_id, instructions=instructions, html_template=html_template)
    return RedirectResponse(f"/brands/{brand_id}/settings", status_code=303)


@router.post("/brands/{brand_id}/history/{content_type}/{entry_id}/delete")
async def delete_history_entry_endpoint(
    brand_id: str, content_type: ContentType, entry_id: int
) -> RedirectResponse:
    _get_brand_or_404(brand_id)
    delete_content_history_entry(brand_id, entry_id)
    return RedirectResponse(f"/brands/{brand_id}/settings", status_code=303)


@router.post("/brands/{brand_id}/history/{content_type}/purge")
async def purge_history_endpoint(brand_id: str, content_type: ContentType) -> RedirectResponse:
    _get_brand_or_404(brand_id)
    purge_content_history(brand_id, content_type)
    return RedirectResponse(f"/brands/{brand_id}/settings", status_code=303)
