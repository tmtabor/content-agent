import logging
import sys
from pathlib import Path
from typing import TextIO

import logfire

from agent.config import settings

DEBUG_LOG_PATH = Path("logs/debug.log")


class _TeeWriter:
    """Duplicates writes to multiple streams — used to mirror console
    output to a file in debug mode without losing the live terminal view.
    """

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        # Logfire's colors="auto" checks this to decide whether to emit ANSI
        # codes — always False here so the mirrored file stays plain,
        # grep-friendly text regardless of what the real terminal supports.
        return False


def configure_logging() -> None:
    """Configure Logfire observability and stdlib logging routing.

    Call this once at application startup before creating any agents.

    If LOGFIRE_TOKEN is set, traces are sent to Logfire cloud.
    If not set, traces are printed to the console (development mode).

    AGENT_DEBUG=true (settings.debug) additionally: prints the full
    prompt/context sent to the model (Logfire's console `verbose` mode,
    off by default), forces DEBUG-level logging regardless of
    AGENT_LOG_LEVEL, and mirrors all of it to logs/debug.log — see
    agent/config.py's `debug` field for why this is a separate switch from
    log_level. Off (the default), behavior is unchanged from before this
    flag existed: console only, nothing written to disk.
    """
    logfire_kwargs: dict = {}

    if settings.logfire_token:
        logfire_kwargs["token"] = settings.logfire_token
    elif settings.debug:
        DEBUG_LOG_PATH.parent.mkdir(exist_ok=True)
        # "a", not "w": with --reload, editing any watched file respawns the
        # worker process, which reimports this module and calls
        # configure_logging() again — "w" would silently truncate the whole
        # session's log on every autoreload, discarding a real testing
        # session out from under whoever's mid-walkthrough (confirmed this
        # happening: four reloads while iterating on unrelated template
        # changes wiped the log three times over). Append survives that;
        # the per-line timestamp is enough to tell restarts apart if needed.
        # buffering=1: line-buffered, so the file is current if read while
        # the server is still running, not just after it exits. Deliberately
        # not a `with` block — this file is meant to stay open for the
        # process lifetime, not a scoped operation.
        log_file = open(DEBUG_LOG_PATH, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        logfire_kwargs["send_to_logfire"] = False
        # Logfire's default scrubbing redacts any value whose content merely
        # *looks* sensitive (substrings like "session" or "secret"), not just
        # values under sensitive-looking keys. Confirmed this silently
        # replaced real newsletter body content with "[Scrubbed due to
        # 'session']" — ordinary TTRPG copy talks about game sessions and
        # plot secrets constantly. Debug mode exists to see the actual
        # output, so scrubbing defeats its purpose here; nothing in this
        # branch leaves the local machine (send_to_logfire=False above).
        logfire_kwargs["scrubbing"] = False
        logfire_kwargs["console"] = logfire.ConsoleOptions(
            min_log_level="debug",
            include_timestamps=True,
            verbose=True,
            colors=False,
            output=_TeeWriter(sys.stdout, log_file),
        )
    else:
        # Console fallback for local development — no token required
        logfire_kwargs["send_to_logfire"] = False
        logfire_kwargs["console"] = logfire.ConsoleOptions(
            min_log_level="debug",
            include_timestamps=True,
        )

    logfire.configure(**logfire_kwargs)

    # Instrument Pydantic AI — automatically traces all agent runs,
    # tool calls, model requests, and validation retries
    logfire.instrument_pydantic_ai()

    # Route stdlib logging through Logfire so third-party library logs
    # (httpx, anthropic SDK, etc.) appear in traces alongside agent spans
    logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])

    # Set root log level from config — AGENT_DEBUG forces DEBUG regardless,
    # since debug mode is only useful if third-party libraries' own
    # DEBUG-level logging (e.g. the OpenAI client's raw request logging)
    # fires too.
    logging.getLogger().setLevel("DEBUG" if settings.debug else settings.log_level.upper())


def get_logger(name: str) -> logging.Logger:
    """Get a stdlib logger that routes through Logfire.

    Usage: logger = get_logger(__name__)
    """
    return logging.getLogger(name)
