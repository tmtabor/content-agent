"""In-memory tracking for background, multi-stage generation jobs.

Local, single-process, single-user app (see agent-web-ui skill's rationale
for this whole stack) — an in-memory dict is enough, no persistent queue
needed. Jobs are looked up by a random token embedded in the polling URL,
not by brand_id, so two browser tabs generating for the same brand at once
don't collide or overwrite each other's progress.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Literal

JobStatus = Literal["running", "done", "error"]


@dataclass
class Job:
    status: JobStatus = "running"
    phase: str = ""
    result: Any = None
    error: str | None = None


_jobs: dict[str, Job] = {}

# asyncio only holds a *weak* reference to a task once nothing else does —
# an unreferenced task can be garbage-collected before it finishes. Since
# run_job() below is fired with asyncio.create_task() and then the request
# handler returns immediately (that's the whole point — don't block on it),
# nothing else would hold a reference without this set.
_background_tasks: set[asyncio.Task] = set()


def create_job() -> str:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = Job()
    return job_id


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs: Any) -> None:
    job = _jobs.get(job_id)
    if job is None:
        return
    for key, value in kwargs.items():
        setattr(job, key, value)


def pop_job(job_id: str) -> Job | None:
    """Remove and return a job — call once its terminal state has been
    rendered so the dict doesn't grow unbounded over a long session.
    """
    return _jobs.pop(job_id, None)


def run_in_background(coro: Any) -> None:
    """Schedule a coroutine to run without blocking the caller, keeping a
    strong reference until it finishes so it can't be garbage-collected.
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
