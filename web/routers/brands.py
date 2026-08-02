"""Brand creation — the "+" tab in the top brand bar."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from db.repository import create_brand
from web.context import base_context
from web.jinja import templates

router = APIRouter()


@router.get("/brands/new", response_class=HTMLResponse)
async def new_brand_form(request: Request) -> HTMLResponse:
    context = base_context(active_brand_id=None, active_agent=None)
    context.update({"request": request, "brand": None})
    return templates.TemplateResponse(request, "brand_form.html", context)


@router.post("/brands")
async def create_brand_endpoint(
    name: str = Form(...),
    background: str = Form(""),
    voice: str = Form(""),
    audience: str = Form(""),
    skypilot_id: str = Form(""),
) -> RedirectResponse:
    brand = create_brand(
        name=name,
        background=background,
        voice=voice,
        audience=audience,
        skypilot_id=skypilot_id,
    )
    return RedirectResponse(f"/brands/{brand.id}/bluesky", status_code=303)
