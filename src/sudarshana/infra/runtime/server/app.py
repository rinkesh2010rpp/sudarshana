"""Run Sudarshana on one machine, without Modal. A test runtime.

    cd src
    python -m sudarshana.infra.runtime.server

That launcher (__main__.py) loads .env first, then calls main() here.

What replaces Modal:
- Storage is a plain folder: /data, or SUDARSHANA_DATA. Commit/reload do nothing.
- The conversation thread is SQLite in that folder, as on Modal.
- The status, dedup and a2a stores live in memory: a restart forgets them.
- Turns run one at a time on one worker thread, the guarantee the Modal app
  gets from max_containers=1. The Telegram "run" dedup gate relies on it.
- The hourly turn runs only with HOURLY=1 (off by default for a test).
- One HTTP server (127.0.0.1:PORT, default 8000) serves the owner's chat and
  the status API (/api/status, /api/events).
- The owner's chat is CHANNEL: "webchat" (default) is a page at /chat in this
  machine's browser, no bot or tunnel needed; "telegram" serves POST
  /telegram for a bot's webhook (put a tunnel in front of it).

There is no turn timeout here (Modal stops a turn at 1500 s); the step cap
in run_turn still stops a runaway.
"""
import os
import queue
import threading
import time
import traceback

from fastapi import FastAPI, Header

from sudarshana.core.agent import CHECKPOINT_DB_PATH, build_agent
from sudarshana.core.config import VOLUME_PATH
from sudarshana.core.deps import Deps
from sudarshana.core.turns import hourly_turn
from sudarshana.infra.storage.local.stores import DictKV, LocalStorage
from sudarshana.infra.storage.sqlite.checkpoints import SqliteCheckpoints
from sudarshana.interfaces.http.status import create_status_app
from sudarshana.interfaces.telegram.send import TelegramNotifier
from sudarshana.interfaces.telegram.turn import telegram_turn
from sudarshana.interfaces.telegram.webhook import handle_update
from sudarshana.interfaces.webchat.chat import WebChat, WebChatNotifier, chat_routes, chat_turn


def make_deps(owner_notifier) -> Deps:
    os.makedirs(VOLUME_PATH, exist_ok=True)
    return Deps(
        storage=LocalStorage(VOLUME_PATH),
        checkpoints=SqliteCheckpoints(CHECKPOINT_DB_PATH),
        status_kv=DictKV(),
        claims_kv=DictKV(),
        a2a_kv=DictKV(),
        owner_notifier=owner_notifier,
    )


def create_app(deps, enqueue, chat: WebChat | None = None) -> FastAPI:
    """The HTTP side. `enqueue(job)` hands a turn to the worker; nothing
    here runs the model, so a request is answered in milliseconds. With a
    `chat`, the web chat is served; without one, the Telegram webhook."""
    web = FastAPI(title="sudarshana-server")

    if chat is not None:
        web.include_router(chat_routes(chat, enqueue))
        web.mount("/", create_status_app(deps))
        return web

    @web.post("/telegram")
    def telegram(payload: dict, x_telegram_bot_api_secret_token: str = Header(default="")):
        return handle_update(
            deps,
            payload,
            x_telegram_bot_api_secret_token,
            lambda text, update_id: enqueue(("telegram", text, update_id)),
        )

    # Everything else is the status API (/api/status, /api/events).
    web.mount("/", create_status_app(deps))
    return web


def run_worker(agent, deps, jobs: queue.Queue, chat: WebChat | None = None) -> None:
    """Run queued turns one at a time, forever. A failed turn is logged and
    the worker carries on, as a failed Modal input does."""
    while True:
        job = jobs.get()
        try:
            if job[0] == "chat":
                chat_turn(agent, deps, chat, job[1])
            elif job[0] == "telegram":
                telegram_turn(agent, deps, job[1], job[2])
            elif job[0] == "hourly":
                hourly_turn(agent, deps)
        except Exception:  # noqa: BLE001 — one bad turn must not stop the worker
            traceback.print_exc()
        finally:
            jobs.task_done()


def run_hourly(jobs: queue.Queue) -> None:
    """Queue an hourly turn at the top of every hour, like the Modal cron."""
    while True:
        now = time.time()
        time.sleep(3600 - now % 3600)
        jobs.put(("hourly",))


def main() -> None:
    import uvicorn

    channel = os.environ.get("CHANNEL", "webchat")
    if channel not in ("webchat", "telegram"):
        raise SystemExit(f"CHANNEL must be webchat or telegram, not {channel!r}")
    chat = WebChat() if channel == "webchat" else None
    deps = make_deps(WebChatNotifier(chat) if chat else TelegramNotifier())
    print(f"[server] workspace {VOLUME_PATH}; building the agent...")
    agent = build_agent(deps)
    jobs: queue.Queue = queue.Queue()
    threading.Thread(target=run_worker, args=(agent, deps, jobs, chat), daemon=True, name="turns").start()
    if os.environ.get("HOURLY") == "1":
        threading.Thread(target=run_hourly, args=(jobs,), daemon=True, name="hourly").start()
        print("[server] hourly turn: on")
    else:
        print("[server] hourly turn: off (set HOURLY=1 to turn it on)")
    port = int(os.environ.get("PORT", "8000"))
    if chat:
        print(f"[server] chat: open http://127.0.0.1:{port}/chat in this machine's browser")
    else:
        print(f"[server] telegram: point the bot's webhook at <tunnel>/telegram (local port {port})")
    uvicorn.run(create_app(deps, jobs.put, chat), host="127.0.0.1", port=port)
