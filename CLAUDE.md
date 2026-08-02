# CLAUDE.md

Content Agent generates Bluesky posts/threads and email newsletters for one
or more configured brands. Built from `agent-template`
(https://github.com/tmtabor/agent-template) but has diverged from several of
its conventions — see "Departures from the base template" below before
assuming template docs/patterns still apply.

## Commands

```bash
uv sync --group dev                          # install deps
uv run pytest                                # unit tests — TestModel, no real model calls
uv run pytest -m eval                        # reliability evals — real model, no API key needed (local Ollama)
uv run ruff check .                          # lint
uv run ruff format .                         # format
uv run uvicorn web.main:app --reload --port 8000   # run the app
```

`asyncio_mode = "auto"` is set in `pyproject.toml` — async tests need no
`@pytest.mark.asyncio` decorator.

## Architecture

```
db/               SQLite layer — the single .db file holding all app state
  schema.sql        CREATE TABLE statements (additive-only after initial launch — see below)
  connection.py      get_connection(), init_db(), additive-migration runner
  models.py          Pydantic models — Brand, {Bluesky,Newsletter}Settings, content payloads
  repository.py       All CRUD — brand/settings/history, FIFO history trim

agent/agents/
  bluesky.py         Bluesky post/thread generation — single call
  newsletter.py       Newsletter generation — split into two sequential calls (see below)

integrations/
  skypilot.py         SkyPilot API client — plain httpx, not an agent tool (see below)

web/                FastAPI + Jinja2 + HTMX, no build step
  main.py             App + lifespan (calls init_db() at startup, not import time)
  jobs.py             In-memory background-job tracking for newsletter's polling UI
  context.py          CONTENT_AGENTS registry (left-nav entries) + shared template context
  routers/            brands.py, bluesky.py, newsletter.py, settings.py
  templates/           base.html (brand tabs + left nav) + per-agent pages + partials/

tests/              Unit tests, TestModel only, no real model calls
evals/              Reliability evals against the real configured model (-m eval)
```

## Departures from the base template

The template assumes **one** agent per app: `scripts/choose_pattern.py`
picks single/supervisor/tool_calling, deletes the other two stubs, and
`agent/agents/__init__.py` re-exports one canonical set of names
(`run_agent`, `AgentOutput`, `AgentDeps`, `agent`). This app needs multiple
independent content agents, so that convention was discarded entirely:

- `single.py`, `supervisor.py`, `tool_calling.py`, and `choose_pattern.py`
  are deleted. `agent/agents/bluesky.py` and `agent/agents/newsletter.py`
  are hand-written, each with its own `*Output`/`*Deps`/`run_*_agent`
  names, both re-exported by name (not aliased to a shared canonical name)
  from `agent/agents/__init__.py`.
- The template's example evals (`evals/test_pass_fail.py`,
  `test_llm_judge.py`) targeted the deleted canonical agent and were
  removed rather than adapted. `evals/judge.py` (LLM-as-judge harness) and
  `evals/conftest.py`/`fixtures/` are kept — generic infrastructure, not
  tied to the old canonical names, available for a future content-quality
  eval.
- `agent/prompts/` and `agent/tools/example.py` (unused by either content
  agent — no `load_prompt()` calls, no registered tools) are removed. If a
  future content agent needs tools, copy the pattern from the template
  directly (https://github.com/tmtabor/agent-template/blob/main/agent/tools/example.py)
  rather than resurrecting these.
- Default model changed to `ollama:gemma4` (local, no API key) — see
  `.env.example`. `tests/conftest.py` sets a dummy `OLLAMA_BASE_URL` before
  import for the same reason it sets dummy provider keys: `Settings`
  validates the configured provider at import time.

## Non-obvious architecture

- **Newsletter generation is two sequential agent calls, not one.** The
  original combined schema (`{body_html, subject_pairs}` in one call)
  occasionally failed against gemma4 with "Please return text or include
  your response in a tool call" — the model drifting off the structured-
  output format entirely on the more complex of the app's two schemas.
  Splitting into a body-only call and a subjects-only call (subjects
  written with the *actual* generated body in context, via
  `dataclasses.replace(deps, body_html=...)`) fixed it — see
  `agent/agents/newsletter.py`'s module docstring. `run_newsletter_agent()`
  takes an optional `on_phase` callback invoked before each call, which is
  how the web layer surfaces "Generating newsletter body" /
  "Generating subject lines" progress without the agent layer knowing
  anything about HTTP or polling.

- **Bluesky's output schema is `list[str]`, not `list[{text: str}]`.**
  Same root cause as above: gemma4 reliably emits a flat string list but
  not a list of single-field wrapper objects. If you're tempted to wrap a
  single field in its own object in either agent's output schema, don't —
  keep output schemas as flat as the data actually requires, especially
  against a small/local model. `evals/test_reliability.py` exists
  specifically to catch a regression here.

- **Both agents use `retries=3` and `model_settings={"temperature": 0.6}`,
  not pydantic-ai's defaults (`retries=1`, gemma4's default temperature of
  1.0).** The default retry budget gives the model exactly one chance to
  self-correct a validation failure before hard-failing the whole
  generation — too tight for a small local model's occasional slips.
  Temperature 1.0 measurably increases how often the model drifts off the
  tool-call format entirely. If you swap to a more capable hosted model,
  these are worth revisiting — they're tuned for gemma4's specific failure
  modes, not universally necessary.

- **SkyPilot posting is a plain `httpx` client, not an agent tool.**
  Sending a post happens after a human reviews and approves the generated
  draft in the UI — it's a deterministic action the web layer takes
  (`web/routers/bluesky.py` calls `integrations/skypilot.py` directly), not
  something the model decides to do. Don't move it into an `@agent.tool` on
  `bluesky_agent`.

- **`db/connection.py`'s migrations are additive-only, no down migrations.**
  `_ADDITIVE_MIGRATIONS` is a list of `(table, column, ddl)` tuples applied
  via `PRAGMA table_info` + `ALTER TABLE ... ADD COLUMN`, idempotent and run
  on every `init_db()` call (including every app startup). There's no
  migration framework — for a schema change beyond adding a nullable/
  defaulted column, you'll need to extend this mechanism or write a one-off
  script; there's nothing here that handles renames or drops.

- **Content history is FIFO-capped at `HISTORY_LIMIT = 20`** per
  `(brand_id, content_type)`, enforced by `repository.add_content_history()`
  deleting anything outside the most-recent 20 immediately after insert —
  not a background job, not lazy cleanup.

- **The web layer keeps no server-side draft/session state.** After
  generation, the result partial embeds the original prompt (and, for
  Bluesky, thread settings) in hidden form fields. Regenerate re-posts
  those hidden fields to the same `/generate` endpoint, discarding any
  edits and never touching history; Send/Mark-as-Used submit whatever is
  currently in the (possibly edited) textareas. The HTML form is the state
  — there's no session cookie or server-side draft store to keep in sync.

- **`web/jobs.py`'s job tracking is in-memory, single-process.** Fine for
  this app (local, single-user, per the whole stack's rationale — see
  `.agents/skills/agent-web-ui/SKILL.md`), but jobs don't survive a reload/
  restart, and `asyncio.create_task()`'s result is deliberately held in a
  module-level set (`_background_tasks`) — asyncio only holds a *weak*
  reference to a task otherwise, and an unreferenced task can be garbage-
  collected before it finishes.

- **`web/main.py` calls `init_db()` from a lifespan handler, not at module
  import time.** Matters for tests: `TestClient` only runs the lifespan
  when used as a context manager (`with TestClient(app) as client:`), which
  is also when `tests/conftest.py`'s `temp_db` fixture has already
  repointed `settings.db_path` at a per-test temp file. Calling `init_db()`
  at bare import time would instead run once, against whatever
  `settings.db_path` was at first import, and every test after that would
  hit an uninitialized temp file.

- **Unit tests never need real credentials or a running Ollama.**
  `tests/conftest.py`'s autouse fixture overrides every `Agent` instance
  found under any loaded `agent.agents` module with `TestModel` — it
  iterates `sys.modules`, so it's agnostic to which/how many agent modules
  exist (no hardcoded name list to keep in sync when a new content agent is
  added).

- **`evals/test_reliability.py` is deliberately strict, not
  flakiness-tolerant.** It asserts all 5 attempts succeed per agent, not
  "at least N%" — an occasional failure here is meant to be read as a real
  reliability regression, not noise to raise the sample size away.

## Adding a third content agent

1. `agent/agents/<name>.py` — its own output schema(s), `*Deps` dataclass,
   `Agent` object(s), `run_<name>_agent()`. Look at whether the data
   genuinely needs one call or benefits from a split like newsletter's.
2. `db/schema.sql` — a `<name>_settings` table if it needs brand-specific
   config beyond free-text instructions (see `bluesky_settings`'s
   `hashtags` column for a precedent); add the corresponding entry to
   `db/connection.py`'s `_ADDITIVE_MIGRATIONS` if the app already has real
   user data (it will, once this ships) rather than relying on
   `CREATE TABLE IF NOT EXISTS` alone.
3. `db/repository.py` — get/update functions for the new settings table;
   reuse `add_content_history()`/`list_content_history()` as-is, just add
   the new content type to `db/models.py`'s `ContentType` literal and
   `_PAYLOAD_MODELS`.
4. `web/routers/<name>.py` + templates — follow `bluesky.py` (synchronous,
   single call) or `newsletter.py` (background job + polling, if the agent
   is slow or multi-stage) as the closer match.
5. Add an entry to `web/context.py`'s `CONTENT_AGENTS` list — that's the
   only place the left-nav needs touching.
6. Re-export the new agent's names from `agent/agents/__init__.py`
   alongside the existing two.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
