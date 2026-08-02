# Content Agent

A local web app that generates on-brand content — Bluesky posts/threads and
email newsletters — for one or more brands. Each brand gets its own
background, voice, and audience context; each content type has its own
generation agent, its own configurable instructions, and its own history so
it doesn't repeat itself. Runs against a local Ollama model by default, no
API key required.

## What it does

- **Multiple brands**, each with its own tab: name, background, voice,
  audience, and SkyPilot account ID.
- **Bluesky agent**: generates a single post or a thread (each post capped
  at 300 characters), using per-brand hashtags and formatting instructions.
  Review and edit before you **Regenerate**, **Send to SkyPilot**
  (immediately or scheduled), or **Mark as Used**.
- **Newsletter agent**: generates an HTML body plus 5 subject/description
  pairs for A/B testing, optionally following a brand-specific HTML
  template. Review and edit before you **Regenerate**, **Copy to
  Clipboard**, or **Mark as Used**.
- **Anti-repetition**: the last 20 pieces of content per brand, per content
  type are kept and fed back into generation so new posts/newsletters don't
  reuse the same jokes or angles.
- **Everything in one SQLite file** — brand config, per-agent settings, and
  content history. Never committed (see `.gitignore`).

## Quickstart

Prerequisites: Python 3.13, [uv](https://docs.astral.sh/uv/), and a local
[Ollama](https://ollama.com) install with the default model pulled:

```bash
ollama pull gemma4
```

Then:

```bash
uv sync --group dev

cp .env.example .env
# Edit .env: OLLAMA_BASE_URL is already set for a local Ollama install.
# Add SKYPILOT_API_KEY if you want "Send to SkyPilot" to work
# (https://skypilot.social/static/html/api.html).

uv run uvicorn web.main:app --reload --port 8000
```

Open <http://localhost:8000> — with no brands yet, it lands on the "add a
brand" form. The app's SQLite file (`content_agent.db` by default) is
created automatically on first run.

## Using it

- **Add a brand**: click the **+** tab, fill in name/background/voice/
  audience/SkyPilot ID.
- **Generate content**: pick a content agent in the left column (Bluesky or
  Newsletter), write a prompt, hit Generate. Newsletter generation runs in
  two stages (body, then subject lines) and shows progress for each.
- **Review**: the draft is fully editable before you act on it.
  Regenerating discards edits and reruns from the original prompt without
  touching history; the other actions save the (possibly edited) content.
- **Brand settings**: click the gear icon on a brand's tab to edit its core
  fields, configure Bluesky hashtags/instructions and newsletter
  instructions/HTML template, browse or purge past content, or delete the
  brand entirely.

## Configuration

All settings are read from `.env` (copy `.env.example` to start). Agent
settings use an `AGENT_` prefix; provider/integration keys keep their
standard names.

| Variable | Default | Notes |
|---|---|---|
| `AGENT_MODEL` | `ollama:gemma4` | Any [pydantic-ai](https://ai.pydantic.dev) model string. Switching to a hosted provider (`anthropic:...`, `openai:...`, `google:...`) needs that provider's API key set too. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Required for the default local model — must include the `/v1` suffix. |
| `AGENT_DB_PATH` | `content_agent.db` | The single SQLite file holding all brand config and content history. |
| `SKYPILOT_API_KEY` | — | Required only for "Send to SkyPilot". |
| `AGENT_LOG_LEVEL` | `INFO` | Standard Python logging level. |
| `LOGFIRE_TOKEN` | unset | If set, traces go to Logfire cloud; otherwise they print to the console. |
| `AGENT_JUDGE_MODEL` | `anthropic:claude-opus-4-8` | Used only by evals that don't ship yet — reserved for future LLM-as-judge content-quality checks. |

## Development

```bash
uv run pytest              # unit tests — TestModel, no real model calls, no API key
uv run pytest -m eval      # reliability evals against the real model, see below
uv run ruff check .        # lint
uv run ruff format .       # format
```

**Reliability evals** (`evals/test_reliability.py`): each agent generates 5
times against the real configured model and every attempt must succeed and
respect its hard constraints (300-char cap, exactly 5 distinct newsletter
subject lines). This exists because a small local model occasionally
produces output a much larger hosted model wouldn't — these evals catch a
reliability regression (from a model swap, prompt edit, or schema change)
before a user hits it. No API key needed against the default local model,
but it does need a real, running Ollama with the model pulled, and takes a
few minutes to run.

See `CLAUDE.md` for architecture notes, including how to add a third
content agent.

## Project structure

```
db/                    SQLite layer — brands, per-agent settings, content history
  schema.sql / connection.py / models.py / repository.py

agent/agents/
  bluesky.py            Bluesky post/thread generation
  newsletter.py          Newsletter generation (body, then subject lines)

integrations/
  skypilot.py            SkyPilot API client (posting/scheduling)

web/                    FastAPI + Jinja2 + HTMX UI
  main.py, routers/, templates/, static/, jobs.py (background-job polling)

tests/                  Unit tests (TestModel, no real model calls)
evals/                  Reliability evals against the real model (-m eval)
```

## License

BSD 3-Clause — see [LICENSE](LICENSE). Built from
[agent-template](https://github.com/tmtabor/agent-template).
