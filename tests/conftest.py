"""Pytest fixtures for unit tests."""

import os

# Unit tests must run with no real credentials and no API calls. Settings
# requires a provider key for the selected model at import time, and the
# module-level Agent construction may create a provider client that also
# wants a key — set dummy values before anything under agent/ is imported.
# setdefault() leaves real keys untouched if they are present. The default
# AGENT_MODEL is ollama:gemma4, which needs OLLAMA_BASE_URL to build (no
# network call happens at import time, so a fake URL is fine here).
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("ANTHROPIC_API_KEY", "unit-test-dummy-key")
os.environ.setdefault("OPENAI_API_KEY", "unit-test-dummy-key")

import importlib  # noqa: E402
import sys  # noqa: E402
from contextlib import ExitStack, suppress  # noqa: E402

import pytest  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from agent.config import settings  # noqa: E402
from agent.logging import configure_logging  # noqa: E402
from db.connection import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Configure logging once for the test session."""
    configure_logging()


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point every test at a fresh, isolated sqlite file.

    db/connection.py reads settings.db_path fresh on every call rather than
    caching it at import time specifically so this works — no test can leak
    state into another test or into a real content_agent.db on disk.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    init_db()
    yield db_path


@pytest.fixture(autouse=True)
def override_all_agents_with_test_model():
    """Safety net: no unit test may ever hit a real model API.

    Overrides every Agent defined under agent.agents — including nested
    worker agents that tools delegate to — with TestModel. Tests can still
    apply their own override on top; the innermost override wins.

    The agent modules are pre-imported here so the override also covers
    modules a test imports lazily in its body — otherwise a module first
    imported mid-test would escape the net.
    """
    for module_name in ("bluesky", "newsletter"):
        with suppress(ModuleNotFoundError):
            importlib.import_module(f"agent.agents.{module_name}")

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("agent.agents") and module is not None:
                for value in vars(module).values():
                    if isinstance(value, Agent):
                        stack.enter_context(value.override(model=TestModel()))
        yield
