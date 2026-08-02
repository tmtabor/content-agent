"""Small render helpers shared across routers."""

from fastapi import Request
from fastapi.responses import HTMLResponse

from web.jinja import templates


def render_error(request: Request, message: str) -> HTMLResponse:
    """Render the error partial — used for htmx-swapped failure states
    (generation errors, SkyPilot errors) so the page doesn't hard-fail.
    """
    return templates.TemplateResponse(request, "partials/error.html", {"message": message})
