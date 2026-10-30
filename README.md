# Sudarshana

An autonomous agent that Rinkesh built and runs on [Modal](https://modal.com).
It wakes up every hour, works one step of its roadmap, writes a daily blog for
the public site ([sudarshana-gateway](https://github.com/rinkesh2010rpp/sudarshana-gateway)),
and answers Rinkesh on Telegram. It has an unsandboxed shell and git access to
its own repos, and it keeps its memory as files on a persistent Volume.

All the code is in [`agent/main.py`](agent/main.py). It is a LangChain
[deepagents](https://github.com/langchain-ai/deepagents) agent on a single
checkpointed conversation thread.

## How it runs

```
Telegram ──► telegram_webhook ──spawn──► process_message ─┐
Cron 0 * * * *  ──► hourly_trigger ──spawn──► hourly_checkin ──┴─► Sudarshana._invoke ─► deep agent
                                                                                 │
                                                                  final message ──► Telegram
```

- **One container (`max_containers=1`).** Telegram and hourly turns
  queue, so two of them never write `state.md` or the day's log at the same
  time. A turn can run for up to 1500 s.
- **Triggers and the webhook call `spawn()` and return right away.** A webhook
  that waited for the work got retried by Telegram, and a cron wrapper that
  waited got preempted and retried. Both started duplicate runs.
- **Python delivers the reply, not a tool call.** `_invoke` always sends the
  agent's final message to Telegram and prints the full trace to the Modal logs.
- **Only `TELEGRAM_ALLOWED_USER_ID` can talk to it.** Messages from anyone else
  are dropped without a reply. `/agents …` commands go straight to the A2A
  controls and never reach the model.
- **What the cycles do is set by prompt, not code:** `SYSTEM_PROMPT`,
  `SYSTEM_PROMPT` and `HOURLY_TASK` in `main.py`.

## The agent

| Piece | What it does |
|---|---|
| Model | OpenRouter through `ChatOpenRouter` (`USE_OPENROUTER=1`, `OPENROUTER_MODEL`). Reasoning is sent back on later calls, and there is no provider pinning. The `ChatOpenAI` branch points at a self-hosted endpoint and is kept only as a fallback. |
| Backend | `LocalShellBackend` rooted at `/data`, with the real environment (so `GITHUB_TOKEN` reaches git). Middleware artifacts also go to the Volume. |
| Memory | `/data/memory/state.md` and the knowledge `index.md` are reloaded into the system message on every turn. |
| Skills | `/data/skills/` is rescanned on every turn. |
| Todos | `write_todos` (`TodoListMiddleware`). |
| Summarization | Compacts the thread at ~70K tokens and keeps the last ~8K. The model profile is capped at 128K input. |
| Checkpointer | `SqliteSaver` at `/data/checkpoints.db`, one thread (`sudarshana`). It checkpoints once per turn. |
| Subagent | The general-purpose `task` subagent, declared with its own summarization, todos and tool screen. |
| Tools | File tools plus `execute`, `search_web` (DuckDuckGo via `ddgs`), `inbox_review` / `inbox_set_status`, and the A2A tools. |
| A2A | [a2a-hotplug](https://github.com/rinkesh2010rpp/a2a-hotplug) adds `add_agent` / `send_agent_task` / `check_agent_task`. Its store is the `sudarshana-a2a-agents` `modal.Dict`, kept alive by the hourly turn. |
| Tool screen | [Taster](https://github.com/rinkesh2010rpp/taster-ai) checks tool results for prompt injection (Jev with a heuristic fallback) when `JEV_TOOL_SCREEN=1`. It enforces on `search_web`, network-touching `execute`, and A2A replies. Rules: `TOOL_SCREEN_DEFAULT_RULES`, or override them with `JEV_TOOL_SCREEN_RULES`. |

Every turn also gets a fresh `Current time:` system message, plus the visitor
inbox intake when there is anything in it.

## Public endpoints

Both are on the same Modal app, with CORS open so the gateway site can call them.

- **`status-api`**: `GET /api/status` and `GET /api/events`. A read-only view of
  the `sudarshana-status` `modal.Dict` (the current turn and recent turn
  history). It mounts no Volume.
- **`inbox-api`**: `POST /api/inbox` lets visitors submit items, and
  `GET /api/inbox` serves the public board. An item stays private (`received`)
  until the agent approves it (`submitted`), then moves through `in_progress`
  to `completed`. `rejected` is a private end state. The data is stored in
  `/data/visitor-inbox.db`.

## What lives on the Volume (`sudarshana-files`, mounted at `/data`)

| Path | Contents |
|---|---|
| `VISION.md`, `ROADMAP.md`, `actions/`, `logs/<date>.md` | The agent's own canonical record. It maintains these, and they win over everything derived. |
| `memory/state.md`, `memory/decisions.md` | Where things stand right now, and the decisions log. |
| `memory/knowledge/` | Compiled knowledge pages. The rules are in [`agent/schema.md`](agent/schema.md). Code regenerates `index.md` and `lint.md`. |
| `skills/` | The skills library. |
| `checkpoints.db` | The conversation thread. |
| `llm-calls/` | One log entry for every model call. |
| `tool-screen/`, `a2a-calls/` | Tool-screen verdicts and A2A call logs (JSONL). |
| `visitor-inbox.db` | The visitor inbox. |
| `sudarshana/`, `sudarshana-gateway/` | The agent's working checkouts of its repos. |

## Configuration

Secrets come from a local `.env` through `modal.Secret.from_dotenv()`. The
code reads these names:

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USER_ID` | The bot, and the only user it answers |
| `USE_OPENROUTER`, `OPENROUTER_MODEL`, `OPENROUTER_API_KEY` | The production model. The key is also used by Jev. |
| `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` | The fallback OpenAI-compatible endpoint |
| `GITHUB_TOKEN` | Git push and `gh` from the agent's shell |
| `JEV_TOOL_SCREEN`, `JEV_TOOL_SCREEN_TIMEOUT_S`, `JEV_TOOL_SCREEN_RULES` | The tool-result screen (off unless set to `1`) |
| `JEV_INBOX_TRIAGE`, `JEV_CONFIDENCE`, `JEV_TIMEOUT_S` | Jev triage of inbox submissions (off unless set to `1`) |

## Deploy

```bash
pip install -r requirements.txt
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal deploy agent/main.py
```

The UTF-8 variables are needed on Windows. Without them, `modal deploy` crashes
while printing its success checkmark.

The deployed webhook URL stays the same across deploys, so you only need to
point Telegram at it once (or again if it changes):

```bash
python agent/set_webhook.py https://<workspace>--sudarshana-sudarshana-telegram-webhook.modal.run
```

`modal serve agent/main.py` runs a temporary copy with its own URL.

### Dependencies

The Modal image installs packages from the `pip_install(...)` list in `main.py`.
**It does not read `requirements.txt`**, so update both when you add a
dependency. Taster and a2a-hotplug are pinned to git tags so that a change in
either can't reach the agent unannounced. Everything else is unpinned.

## Debugging

- `modal app logs sudarshana` shows the full trace of each turn and the
  `[timing]` lines, and it includes past output.
- `modal volume get sudarshana-files llm-calls` downloads the per-call model logs.
- `GET …status-api…/api/status` shows whether a turn is running right now.
