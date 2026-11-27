# Sudarshana

An autonomous agent that Rinkesh built and runs on [Modal](https://modal.com).
It wakes up every hour, works one step of its roadmap, writes a daily blog for
the public site ([sudarshana-gateway](https://github.com/rinkesh2010rpp/sudarshana-gateway)),
and answers Rinkesh on Telegram. It has an unsandboxed shell and git access to
its own repos, and it keeps its memory as files on a persistent Volume.

The code is the `sudarshana` package in [`src/sudarshana/`](src/sudarshana/); the Modal
wiring is [`infra/runtime/modal/app.py`](src/sudarshana/infra/runtime/modal/app.py). It is a LangChain
[deepagents](https://github.com/langchain-ai/deepagents) agent on a single
checkpointed conversation thread.

## How it runs

```
Telegram ──► telegram_webhook ──spawn──► process_message ─┐
Cron 0 * * * *  ──► hourly_trigger ──spawn──► hourly_checkin ──┴─► run_turn ─► deep agent
                                                                                 │
                                                                  final message ──► Telegram
```

- **One container (`max_containers=1`) for the turns.** Telegram and hourly
  turns queue, so two of them never write `state.md` or the day's log at the
  same time. A turn can run for up to 1500 s.
- **The webhook is not in that container.** `telegram_webhook` is its own
  function — no Volume, no model — so it acks in milliseconds even mid-turn.
  As a method it shared the single container, a delivery that arrived during a
  long turn waited past Telegram's delivery timeout, and each resend started
  another turn (one message became four on 2026-11-11).
- **Resent updates are dropped by `update_id`.** The webhook records each id
  in a rolling window in the `sudarshana-telegram-claims` Dict and acks a
  repeat without spawning. `process_message` holds a second claim for the same
  id: turns are serialised by the single container, so even the pair that races
  past the webhook can't both run.
- **The id is recorded after the spawn, not before.** A spawn that raises
  answers 5xx so Telegram retries; recording first would make that retry look
  like a duplicate and lose the message for good. Claim failures never raise
  either — a Dict outage costs at most one extra turn, a dropped message costs
  everything. The webhook's gate is an optimisation; the serialised "run" gate
  is the guarantee.
- **Triggers call `spawn()` and return right away.** A webhook that waited for
  the work got retried by Telegram, and a cron wrapper that waited got
  preempted and retried. Both started duplicate runs.
- **Python delivers the reply, not a tool call.** `run_turn` returns the
  agent's final message and the caller always sends it: a Telegram turn back to
  Telegram, an hourly turn to the owner's notifier (also Telegram). The full
  trace goes to the Modal logs.
- **Only `TELEGRAM_ALLOWED_USER_ID` can talk to it.** Messages from anyone else
  are dropped without a reply. `/agents …` commands are handled inside
  `process_message` and never reach the model. The id check is defence in depth
  behind `TELEGRAM_WEBHOOK_SECRET`: the webhook first rejects any request that
  doesn't carry Telegram's secret-token header, so a forged update (the URL and
  the id are both discoverable) never reaches the id check or the model.
- **What the cycles do is set by prompt, not code:** `SYSTEM_PROMPT`,
  `SYSTEM_PROMPT` and `HOURLY_TASK` in `core/prompt.py`.

## The agent

| Piece | What it does |
|---|---|
| Model | OpenRouter through `ChatOpenRouter` (`USE_OPENROUTER=1`, `OPENROUTER_MODEL`). Reasoning is sent back on later calls, and there is no provider pinning. The `ChatOpenAI` branch points at a self-hosted endpoint and is kept only as a fallback. |
| Backend | `LocalShellBackend` rooted at `/data`. The shell env is `os.environ` minus the model/telegram/webhook secrets (`SHELL_WITHHELD_SECRETS`); `GITHUB_TOKEN` and system vars stay. This only blocks a naive env dump — the parent process still holds the secrets (readable via `/proc`), so it is not isolation. Middleware artifacts also go to the Volume. |
| Memory | `/data/memory/state.md` and the knowledge `index.md` are reloaded into the system message on every turn. |
| Skills | `/data/skills/` is rescanned on every turn. |
| Todos | `write_todos` (`TodoListMiddleware`). |
| Summarization | Compacts the thread at ~70K tokens and keeps the last ~8K. The model profile is capped at 128K input. |
| Checkpointer | `SqliteSaver` at `/data/checkpoints.db`, one thread (`sudarshana`). It checkpoints once per turn. |
| Subagent | The general-purpose `task` subagent, declared with its own summarization, todos and tool screen. |
| Tools | File tools plus `execute`, `search_web` (DuckDuckGo via `ddgs`), `inbox_review` / `inbox_set_status`, the A2A tools, and `broker_register` / `broker_request` for outside accounts through `credential-broker/`. |
| A2A | [a2a-hotplug](https://github.com/rinkesh2010rpp/a2a-hotplug) adds `add_agent` / `send_agent_task` / `check_agent_task`. Its store is the `sudarshana-a2a-agents` `modal.Dict`, kept alive by the hourly turn. |
| Tool screen | [Taster](https://github.com/rinkesh2010rpp/taster-ai) checks tool results for prompt injection (Jev with a pattern-matching fallback) when `JEV_TOOL_SCREEN=1`. It enforces on `search_web`, network-touching `execute`, and A2A replies (fail-closed: `on_error: block`), and runs a free pattern scan on every other result in shadow. Scope: `TOOL_SCREEN_DEFAULT_SCOPE`, or override it with `JEV_TOOL_SCREEN_RULES` (JSON, same shape). The same switch checks every tool call against `TOOL_CONSTITUTION` (`CONSTITUTION_SCOPE`): enforced on calls to external agents, shadow on everything else — logged, flip `execute` to enforce after reviewing the logs. |

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
| `TELEGRAM_WEBHOOK_SECRET` | Shared secret proving a webhook request came from Telegram; forgeries are rejected before any work runs |
| `USE_OPENROUTER`, `OPENROUTER_MODEL`, `OPENROUTER_API_KEY` | The production model. The key is also used by Jev. |
| `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` | The fallback OpenAI-compatible endpoint |
| `GITHUB_TOKEN` | Git push and `gh` from the agent's shell |
| `BROKER_URL`, `BROKER_TOKEN_ID`, `BROKER_TOKEN_SECRET` | The credential broker's endpoint and proxy token, used by the broker tools; withheld from the shell |
| `JEV_TOOL_SCREEN`, `JEV_TOOL_SCREEN_TIMEOUT_S`, `JEV_TOOL_SCREEN_RULES` | The tool-result screen (off unless set to `1`) |
| `JEV_INBOX_TRIAGE`, `JEV_CONFIDENCE`, `JEV_TIMEOUT_S` | Jev triage of inbox submissions (off unless set to `1`) |

## Deploy

```bash
pip install -r requirements.txt
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal deploy src/sudarshana/infra/runtime/modal/app.py
```

The UTF-8 variables are needed on Windows. Without them, `modal deploy` crashes
while printing its success checkmark.

The deployed webhook URL stays the same across deploys, so you only need to
point Telegram at it once (or again if it changes). The label
`TELEGRAM_WEBHOOK_LABEL` in `app.py` is pinned to the slug the endpoint had
when it was a class method, so moving it out of the class did not change the
URL and Telegram needed no re-registration:

```bash
python scripts/set_webhook.py https://<workspace>--sudarshana-sudarshana-telegram-webhook.modal.run
```

`set_webhook.py` registers `TELEGRAM_WEBHOOK_SECRET` with Telegram as the
webhook's `secret_token`. **Register the secret (step above) before deploying
the code that checks it.** Two failure modes if the order slips: with the secret
*unset* in the deployed env, the webhook answers every update `503` and Telegram
queues them; with the secret *set* but not yet registered, Telegram sends the
wrong token and genuine updates are dropped silently (`200`) rather than queued.
So after changing the secret in `.env`, re-run `set_webhook.py` first, then
redeploy, and send yourself a message to confirm delivery.

`modal serve src/sudarshana/infra/runtime/modal/app.py` runs a temporary copy with its own URL.

### Run locally

`scripts/run_local.py` runs turns on this machine with no Modal: the workspace
is the folder in `SUDARSHANA_DATA`, the status/dedup/a2a stores live in memory,
and owner messages print to the console. The agent's shell is unsandboxed on
your machine, so read the warning at the top of the script first.

### Dependencies

The Modal image installs packages from the `pip_install(...)` list in `app.py`.
**It does not read `requirements.txt`**, so update both when you add a
dependency. Taster and a2a-hotplug are pinned to git tags so that a change in
either can't reach the agent unannounced. Everything else is unpinned.

## Debugging

- `modal app logs sudarshana` shows the full trace of each turn and the
  `[timing]` lines, and it includes past output.
- `modal volume get sudarshana-files llm-calls` downloads the per-call model logs.
- `GET …status-api…/api/status` shows whether a turn is running right now.
