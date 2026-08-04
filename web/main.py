"""FastAPI web UI entry point.

Run with: uv run uvicorn web.main:app --reload --port 8000
"""

import asyncio
import os
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from agent.config import settings as app_settings
from agent.logging import configure_logging, get_logger
from db import init_db
from db.repository import list_brands
from web.jobs import run_in_background
from web.routers import bluesky, brands, linkedin, linkedin_to_bluesky, newsletter, settings

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Deferred to startup rather than run at import time: settings.db_path
    # can still be overridden (e.g. tests pointing it at a temp file) up
    # until the app actually starts, but a plain module-level call would run
    # once against whatever path was current at first import of this module.
    init_db()
    yield


app = FastAPI(title="Content Agent", lifespan=lifespan)

# Traces HTTP requests as spans alongside the agent-run spans configure_logging()
# already sets up, so a full request -> agent.run() trace shows up as one
# connected trace in Logfire (or the console, if no token is set).
logfire.instrument_fastapi(app)


@app.middleware("http")
async def log_request_body_in_debug_mode(request: Request, call_next):
    """Only acts when AGENT_DEBUG=true (see agent/config.py, agent/logging.py)
    — logs the raw form body of every POST/PUT request, the "what did the
    user actually submit" half of debug mode (the other half, what the
    model produced, is logged in each agent/agents/*.py run_* wrapper).
    Registered unconditionally rather than only when debug is on, to avoid
    conditionally wiring the middleware stack for a setting that can't
    change after startup anyway. Starlette caches the body after this
    read, so route handlers' own Form(...)/request.form() parsing
    downstream is unaffected.
    """
    if app_settings.debug and request.method in ("POST", "PUT"):
        body = await request.body()
        logger.debug(
            "Request body",
            extra={
                "method": request.method,
                "path": request.url.path,
                "body": body.decode(errors="replace"),
            },
        )
    return await call_next(request)


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.include_router(brands.router)
app.include_router(bluesky.router)
app.include_router(newsletter.router)
app.include_router(linkedin.router)
app.include_router(linkedin_to_bluesky.router)
app.include_router(settings.router)


@app.get("/")
async def index() -> RedirectResponse:
    existing = list_brands()
    if not existing:
        return RedirectResponse("/brands/new")
    return RedirectResponse(f"/brands/{existing[0].id}/bluesky")


@app.post("/shutdown", response_class=HTMLResponse)
async def shutdown() -> HTMLResponse:
    """Stop the server from the UI — an alternative to Ctrl+C in the
    macOS launcher's terminal window (see scripts/build_launcher.sh).
    """

    async def _stop() -> None:
        await asyncio.sleep(0.3)  # let this response finish sending first
        os.kill(os.getpid(), signal.SIGINT)  # same signal Ctrl+C sends; uvicorn shuts down cleanly

    run_in_background(_stop())
    return HTMLResponse("<p>Shutting down — you can close this tab and the terminal window.</p>")
