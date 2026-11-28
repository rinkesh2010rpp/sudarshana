"""Run Sudarshana on one ordinary machine, without Modal.

    bash src/sudarshana/infra/runtime/server/start.sh    # in the background
    cd src && python -m sudarshana.infra.runtime.server  # in the foreground

__main__.py loads .env first, then calls main() here. The Modal app
(infra/runtime/modal/app.py) is the production runtime; this one runs the
same agent, turns and interfaces on a single process.

What replaces Modal:
- Storage is a plain folder: /data, or SUDARSHANA_DATA. Commit/reload do nothing.
- The conversation thread is SQLite in that folder, as on Modal.
- The status, dedup and a2a stores live in memory: a restart forgets them.
- Turns run one at a time on one worker thread, the guarantee the Modal app
  gets from max_containers=1. The Telegram "run" dedup gate relies on it.
- The scheduled turn runs only with HOURLY=1 (off by default), every
  WAKE_EVERY_MIN minutes (60 by default, at least 5), on the clock: 30 means
  :00 and :30.
- One HTTP server (127.0.0.1:PORT, default 8000) serves the owner's chat and
  the status API (/api/status, /api/events).
- The owner's chat is CHANNEL: "webchat" (default) is a page at /chat in this
  machine's browser behind WEBCHAT_PASSWORD (reach it over an SSH tunnel);
  "telegram" serves POST /telegram for a bot's webhook (put a tunnel in front
  of it, and use a different bot from production's).

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
    """Ports for one machine: the workspace is a local folder and the shared
    stores live in this process."""
    os.makedirs(VOLUME_PATH, exist_ok=True)
    return Deps(
        storage=LocalStorage(VOLUME_PATH),
        checkpoints=SqliteCheckpoints(CHECKPOINT_DB_PATH),
        status_kv=DictKV(),
        claims_kv=DictKV(),
        a2a_kv=DictKV(),
        owner_notifier=owner_notifier,
    )


def create_app(deps, enqueue, chat: WebChat | None = None, password: str = "") -> FastAPI:
    """The HTTP side. `enqueue(job)` hands a turn to the worker; nothing
    here runs the model, so a request is answered in milliseconds. With a
    `chat`, the web chat is served; without one, the Telegram webhook."""
    web = FastAPI(title="sudarshana-server")

    if chat is not None:
        if not password:
            raise ValueError("the web chat needs a password")
        web.include_router(chat_routes(chat, enqueue, password))
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
            else:
                print(f"[server] unknown job {job[0]!r}, skipped")
        except Exception:  # noqa: BLE001 — one bad turn must not stop the worker
            traceback.print_exc()
        finally:
            jobs.task_done()


def wake_every_min() -> int:
    """WAKE_EVERY_MIN, the minutes between scheduled turns: 60 unless set,
    never under 5, so a typo can't turn the agent into a busy loop."""
    raw = os.environ.get("WAKE_EVERY_MIN", "60")
    try:
        minutes = int(raw)
    except ValueError:
        raise SystemExit(f"WAKE_EVERY_MIN must be a whole number of minutes, not {raw!r}")
    if minutes < 5:
        raise SystemExit(f"WAKE_EVERY_MIN must be at least 5, not {minutes}")
    return minutes


def seconds_to_next_wake(now: float, every_min: int) -> float:
    """Seconds until the next multiple of `every_min` minutes on the clock,
    so 60 wakes on the hour and 30 on the hour and the half hour."""
    period = every_min * 60
    return period - now % period


def run_hourly(jobs: queue.Queue, every_min: int = 60) -> None:
    """Queue the scheduled turn every `every_min` minutes, on the clock, like
    the Modal cron (which is hourly)."""
    while True:
        time.sleep(seconds_to_next_wake(time.time(), every_min))
        jobs.put(("hourly",))


def main() -> None:
    import uvicorn

    channel = os.environ.get("CHANNEL", "webchat")
    if channel not in ("webchat", "telegram"):
        raise SystemExit(f"CHANNEL must be webchat or telegram, not {channel!r}")
    chat = WebChat() if channel == "webchat" else None
    password = os.environ.get("WEBCHAT_PASSWORD", "")
    if chat and not password:
        raise SystemExit("[server] no WEBCHAT_PASSWORD configured: set it in .env to use the web chat")
    deps = make_deps(WebChatNotifier(chat) if chat else TelegramNotifier())
    print(f"[server] workspace {VOLUME_PATH}; building the agent...")
    agent = build_agent(deps)
    jobs: queue.Queue = queue.Queue()
    threading.Thread(target=run_worker, args=(agent, deps, jobs, chat), daemon=True, name="turns").start()
    if os.environ.get("HOURLY") == "1":
        every = wake_every_min()
        threading.Thread(target=run_hourly, args=(jobs, every), daemon=True, name="hourly").start()
        print(f"[server] scheduled turn: on, every {every} min")
    else:
        print("[server] scheduled turn: off (set HOURLY=1 to turn it on)")
    port = int(os.environ.get("PORT", "8000"))
    if chat:
        print(f"[server] chat: open http://127.0.0.1:{port}/chat in this machine's browser"
              " (password: WEBCHAT_PASSWORD)")
    else:
        print(f"[server] telegram: point the bot's webhook at <tunnel>/telegram (local port {port})")
    uvicorn.run(create_app(deps, jobs.put, chat, password), host="127.0.0.1", port=port)
