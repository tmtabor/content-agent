"""FastAPI web UI entry point.

Run with: uv run uvicorn web.main:app --reload --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from agent.logging import configure_logging
from db import init_db
from db.repository import list_brands
from web.routers import bluesky, brands, newsletter, settings

configure_logging()


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

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.include_router(brands.router)
app.include_router(bluesky.router)
app.include_router(newsletter.router)
app.include_router(settings.router)


@app.get("/")
async def index() -> RedirectResponse:
    existing = list_brands()
    if not existing:
        return RedirectResponse("/brands/new")
    return RedirectResponse(f"/brands/{existing[0].id}/bluesky")
