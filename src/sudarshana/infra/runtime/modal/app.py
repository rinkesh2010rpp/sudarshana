"""
Sudarshana — a message- and schedule-triggered agent with unsandboxed
shell + git access to its own repo.

Telegram posts to a Modal webhook — its own lightweight function, not a
method — which verifies the sender, drops updates it has already seen
(Telegram resends anything it thinks timed out), spawns the real work and
returns immediately, and runs the message through a LangChain "deep
agent": a model with write_todos plus a LocalShellBackend rooted at a
persistent Modal Volume (file tools + execute_command).

Continuity comes from files the agent maintains on the Volume:
VISION.md, ROADMAP.md, actions/<id>.md, logs/<date>.md.
Every turn (Telegram and scheduled) also runs on one checkpointed
conversation thread (SqliteSaver on the Volume) that keeps the full
history — tool calls and results included — and is condensed by
deepagents' built-in summarization when it nears the context limit.

Delivery is handled by Python, not a model tool call (which the model
sometimes forgot): run_turn returns the agent's final message, the caller
sends it (a Telegram turn back to Telegram, an hourly turn to the owner's
notifier), and the full message trace is printed to the modal logs.

External A2A agents (find, register, call at runtime) come from the
a2a-hotplug package; Rinkesh controls them with /agents on Telegram.
The web endpoints share the app: telegram-webhook (Telegram's door),
status-api (read-only turn status), inbox-api (the visitor inbox) and
control-api (`/control`). See README.md.

This file is wiring only: Modal decorators, the image, and make_deps().
The agent is built once per container in Sudarshana.setup() (build_agent
in core/agent.py); process_message (spawned by telegram_webhook) and
hourly_checkin run turns on it with run_turn. hourly_trigger is a bare wrapper because Modal only
accepts schedule= on @app.function(), not @modal.method(). Change the
hourly behaviour via HOURLY_TASK / the prompt, not code.

    modal serve src/sudarshana/infra/runtime/modal/app.py     # temporary URL
    modal deploy src/sudarshana/infra/runtime/modal/app.py    # stable URL
    python scripts/set_webhook.py <printed-url>
"""

from pathlib import Path

import modal

from sudarshana.core.agent import CHECKPOINT_DB_PATH, build_agent, run_turn
from sudarshana.core.config import VOLUME_PATH
from sudarshana.core.deps import Deps
from sudarshana.core.prompt import HOURLY_TASK
from sudarshana.infra.storage.modal.kv import modal_kv
from sudarshana.infra.storage.modal.volume import ModalStorage, volume
from sudarshana.infra.storage.sqlite.checkpoints import SqliteCheckpoints
from sudarshana.integrations.a2a import A2A_DICT_NAME, _a2a_keep_alive, _agents_command
from sudarshana.interfaces.telegram.send import TelegramNotifier
from sudarshana.interfaces.telegram.webhook import (
    TELEGRAM_CLAIM_DICT_NAME,
    _telegram_record,
    _telegram_seen,
    handle_update,
)
from sudarshana.memory.asks import ASKS_PATH
from sudarshana.memory.inbox import (
    INBOX_MAX_ITEM_LEN,
    INBOX_MAX_NAME_LEN,
    INBOX_PUBLIC_STATUSES,
    _InboxDB,
    _inbox_set_status,
    _inbox_ts,
)
from sudarshana.memory.report import _report_latest, _report_with_progress
from sudarshana.memory.roadmap import (
    ACTIONS_ARCHIVE_DIR,
    ROADMAP_ID_RE,
    TASK_OPEN,
    _rm_find,
    _rm_sorted,
    _roadmap_focus,
    _roadmap_load,
)
from sudarshana.memory.status import (
    STATUS_DICT_NAME,
    STATUS_HISTORY_LIMIT,
    _status_is_zombie,
    _status_parse_ts,
    _status_ts,
    _status_turn_end,
    _status_turn_start,
)
from sudarshana.safety.jev_triage import (
    JEV_CONFIDENCE,
    JEV_ENABLED,
    JEV_VERDICTS,
    _jev_input_hash,
    _jev_triage,
)

app = modal.App("sudarshana")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "gnupg", "ca-certificates", "gh")
    # Keep in sync with requirements.txt.
    .pip_install(
        "requests",
        "fastapi[standard]",
        "deepagents>=0.7.22",
        "langchain-openai",
        "langchain-openrouter",
        "ddgs",
        "langgraph-checkpoint-sqlite",
        # Pinned to a tag so a Taster change can't reach the agent unannounced.
        "taster-ai[langchain] @ git+https://github.com/rinkesh2010rpp/taster-ai@v0.2.0.dev0",
        # External A2A agents. Pinned to a tag, like Taster; it pins a2a-sdk.
        "a2a-hotplug @ git+https://github.com/rinkesh2010rpp/a2a-hotplug@v0.1.0",
    )
    # Last, so editing the script doesn't reinstall the packages above.
    # Not /tmp: copying a file there breaks that folder's permissions, and apt needs it.
    .add_local_file(Path(__file__).parent / "image_setup.sh", "/opt/image_setup.sh", copy=True)
    .run_commands("bash /opt/image_setup.sh")
)


def make_deps() -> Deps:
    """The real ports, built once per container and passed explicitly.
    Every handle is lazy, so building them costs no round trip."""
    return Deps(
        storage=ModalStorage(volume, VOLUME_PATH),
        checkpoints=SqliteCheckpoints(CHECKPOINT_DB_PATH),
        status_kv=modal_kv(STATUS_DICT_NAME),
        claims_kv=modal_kv(TELEGRAM_CLAIM_DICT_NAME),
        a2a_kv=modal_kv(A2A_DICT_NAME),
        owner_notifier=TelegramNotifier(),
    )


@app.cls(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_PATH: volume},
    # Room for a couple of slow reasoning calls (up to 600s each) in one turn.
    timeout=1500,
    # One container, so hourly/Telegram turns queue instead of running
    # side by side and clobbering each other's writes to state.md and the log.
    max_containers=1,
)
class Sudarshana:
    @modal.enter()
    def setup(self):
        # Runs once per container start; self.agent is reused by every
        # turn this container handles afterward.
        self.deps = make_deps()
        self.agent = build_agent(self.deps)

    @modal.method()
    def process_message(self, text: str, update_id: int | None = None):
        import time

        # Second dedup gate (the webhook holds the first). Method calls are
        # serialised by this class's single container, so two copies of one
        # update cannot both pass here — not even the pair that races past the
        # webhook's claim. That serialisation is why this gate is the real
        # guarantee: read-then-write here cannot interleave with another copy.
        if update_id is not None:
            if _telegram_seen(self.deps.claims_kv, update_id, "run"):
                print(f"[telegram] duplicate update_id {update_id} — no turn started")
                return
            # Recorded before the turn so a crash mid-turn can't re-run it.
            # Both helpers swallow Dict errors: this turn was already acked to
            # Telegram, so raising here would discard the message silently.
            _telegram_record(self.deps.claims_kv, update_id, "run")

        # Rinkesh's external-agent controls run here, never through the model.
        if text.strip().lower().split()[:1] == ["/agents"]:
            TelegramNotifier().send(_agents_command(self.agent.a2a, text))
            return

        started = time.monotonic()
        print(f"[timing] process_message started: {text[:200]!r}")

        # Before any model work, so a turn that dies mid-run leaves a trace.
        _turn_key = _status_turn_start(self.deps, "running-telegram")

        result = run_turn(self.agent, text, "telegram", _turn_key)
        # The message came in on Telegram, so the reply goes back there.
        TelegramNotifier().send(result.reply)

        print(f"[timing] process_message finished in {time.monotonic() - started:.1f}s")
        _status_turn_end(
            self.deps,
            _turn_key,
            result.outcome,
            "telegram turn finished",
        )

        # Commit explicitly — the container may be torn down before the
        # background commit timer catches these writes.
        self.deps.storage.commit()

    @modal.method()
    def hourly_checkin(self):
        import time

        started = time.monotonic()
        print("[timing] hourly_checkin started")

        _turn_key = _status_turn_start(self.deps, "running-hourly")
        _a2a_keep_alive(self.deps.a2a_kv)
        result = run_turn(self.agent, HOURLY_TASK, "hourly", _turn_key)
        # Nobody sent this turn a message; its report goes to the owner.
        self.deps.owner_notifier.send(result.reply)

        print(f"[timing] hourly_checkin finished in {time.monotonic() - started:.1f}s")
        _status_turn_end(
            self.deps,
            _turn_key,
            result.outcome,
            "hourly turn finished",
        )
        self.deps.storage.commit()


# --- Telegram intake ---------------------------------------------------------
# The webhook lives OUTSIDE the Sudarshana class on purpose. As a method it
# shared the class's single container (max_containers=1), so a delivery that
# arrived mid-turn waited out the whole turn; Telegram gave up on the HTTP
# request and resent the update, and each resend started another turn — one
# message became four identical turns on 2026-11-11. Here it mounts no Volume,
# never touches the model, and answers in milliseconds; the container limit
# stays on the class, where the turns are.
#
# The label is pinned to the slug the old method endpoint had — the URL
# Telegram is already registered against (`getWebhookInfo` on 2026-11-11:
# https://rinkesh2010rpp--sudarshana-sudarshana-telegram-webhook.modal.run,
# last error "Read timeout expired" at 21:14 PST, mid-turn). A clean rename is
# a separate change that must include re-running scripts/set_webhook.py; until
# then this doubled name is what keeps messages from 404ing after deploy.
TELEGRAM_WEBHOOK_LABEL = "sudarshana-sudarshana-telegram-webhook"

from fastapi import Header  # noqa: E402


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    # Short: this function's whole job is to ack and spawn.
    timeout=60,
    # Unbounded, a burst of retries from Telegram would each spin up a
    # full-image container. Four acks at a time is far more than this needs.
    max_containers=4,
)
@modal.fastapi_endpoint(method="POST", label=TELEGRAM_WEBHOOK_LABEL)
def telegram_webhook(
    payload: dict,
    x_telegram_bot_api_secret_token: str = Header(default=""),
):
    """Telegram's door. The checks, dedup and error policy are in
    interfaces/telegram/webhook.py (handle_update)."""
    return handle_update(
        make_deps(),
        payload,
        x_telegram_bot_api_secret_token,
        lambda text, update_id: Sudarshana().process_message.spawn(text, update_id),
    )


@app.function(image=image)
@modal.asgi_app(label="status-api")
def status_api():
    """status-center tier 1: read-only /api/status + /api/events.

    A separate web endpoint on the same Modal app, reading the shared
    sudarshana-status Dict live (mounts NO volume). Public + read-only like the
    gateway blog: every payload is own-work summaries only, no secrets or
    infra details. CORS is enabled via explicit middleware so the gateway SPA
    can fetch cross-origin.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    deps = make_deps()
    web_app = FastAPI(title="sudarshana-status")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def _snapshot():
        d = deps.status_kv
        current = d.get("current", {}) or {}
        history = d.get("history", []) or []
        last = history[0] if history else None
        # Public API: strip internal keys like _turn_key.
        current = {k: v for k, v in current.items() if not k.startswith("_")}
        return {
            "generated_at": _status_ts(),
            "current": current,
            "last_turn": last,
            "history_count": len(history),
        }

    @web_app.get("/api/status")
    def api_status():
        return _snapshot()

    @web_app.get("/api/events")
    def api_events(limit: int = 50):
        d = deps.status_kv
        history = d.get("history", []) or []
        limit = max(1, min(int(limit), STATUS_HISTORY_LIMIT))
        return {
            "generated_at": _status_ts(),
            "events": history[:limit],
            "current_trace": d.get("current_trace", []) or [],
        }

    return web_app


@app.function(image=image, volumes={VOLUME_PATH: volume}, secrets=[modal.Secret.from_dotenv()])
@modal.asgi_app(label="inbox-api")
def inbox_api():
    """Visitor inbox endpoint.

    POST /api/inbox stores a submission (text + optional name) as 'received'
    (PRIVATE) until it passes the policy check. GET /api/inbox serves the
    public board. CORS is enabled so the gateway SPA can call it cross-origin.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uuid as _uuid

    deps = make_deps()
    web_app = FastAPI(title="sudarshana-inbox")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    class InboxSubmission(BaseModel):
        text: str = Field(..., min_length=1, max_length=INBOX_MAX_ITEM_LEN)
        name: str = Field(default="", max_length=INBOX_MAX_NAME_LEN)

    @web_app.post("/api/inbox")
    def submit(sub: InboxSubmission):
        text = sub.text.strip()
        name = sub.name.strip() or "anonymous"
        if not text:
            return {"ok": False, "error": "empty-item"}
        db = _InboxDB()
        ts = _inbox_ts()
        item_id = f"item:{_uuid.uuid4().hex[:12]}"
        rec = {
            "id": item_id,
            "name": name,
            "text": text,
            "status": "received",
            "created_at": ts,
            "updated_at": ts,
            "artifact_link": None,
        }
        db.insert(rec)
        db.log_event(
            item_id,
            from_status="-",
            to_status="received",
            note=f"name={name}",
            artifact_link="",
        )
        # Explicit commit: the container is torn down before auto-flush.
        deps.storage.commit()

        # Apply a confident Jev verdict now via _inbox_set_status; anything
        # else leaves the item 'received' for the agent's next-cycle review.
        triage = None
        outcome = None  # set only when a transition actually landed
        if JEV_ENABLED:
            triage = _jev_triage(text)
            verdict, confidence = triage.get("verdict"), triage.get("confidence", 0.0)
            if triage.get("ran") and verdict and confidence >= JEV_CONFIDENCE and verdict in JEV_VERDICTS:
                target = JEV_VERDICTS[verdict]
                applied = _inbox_set_status(
                    deps,
                    item_id,
                    target,
                    note=(
                        f"Jev triager: verdict={verdict} confidence={confidence:.2f} "
                        f"probs={triage.get('probabilities')} input_hash={_jev_input_hash(text)}"
                    ),
                )
                if not applied.get("ok"):
                    # Didn't land: log it and fall through to the 'received'
                    # response, so the visitor isn't told otherwise.
                    db.log_event(
                        item_id,
                        from_status="received",
                        to_status="received",
                        note=(
                            f"Jev triager: transition to {target} FAILED "
                            f"({applied.get('error') or 'unknown'}) — leaving "
                            f"received for manual review. "
                            f"input_hash={_jev_input_hash(text)}"
                        ),
                        artifact_link="",
                    )
                    deps.storage.commit()
                else:
                    outcome = target
            else:
                # No auto-action: a 'hold' verdict, low confidence, or a Jev
                # failure. Logged so the reason is auditable.
                if triage.get("ran") and verdict == "hold":
                    reason = f"hold-verdict:{confidence:.2f}"
                elif triage.get("ran"):
                    reason = f"low-confidence:{confidence:.2f}"
                else:
                    reason = triage.get("error") or "unknown"
                db.log_event(
                    item_id,
                    from_status="received",
                    to_status="received",
                    note=f"Jev triager: no auto-action ({reason}) input_hash={_jev_input_hash(text)}",
                    artifact_link="",
                )
                deps.storage.commit()

        if outcome == "submitted":
            return {
                "ok": True,
                "id": item_id,
                "status": "submitted",
                "message": (
                    "Received and added to the public queue — it meets the "
                    "moderation policy. I'll get to it as soon as I can."
                ),
            }
        if outcome == "rejected":
            return {
                "ok": True,
                "id": item_id,
                "status": "rejected",
                "message": (
                    "Not accepted — thanks for writing. This queue only carries "
                    "requests that meet the moderation policy."
                ),
            }
        return {
            "ok": True,
            "id": item_id,
            "status": "received",
            "message": (
                "Received — I'll review this on my next run, and if it meets "
                "policy it'll be added to the queue automatically."
            ),
        }

    @web_app.get("/api/inbox")
    def read(limit: int = 50):
        db = _InboxDB()
        rows = db.fetch(INBOX_PUBLIC_STATUSES)
        rows.sort(key=lambda r: r["created_at"] or "", reverse=True)
        rows = rows[: max(1, min(int(limit), 100))]
        items = []
        for r in rows:
            ans = db.get_answer(r["id"]) if r["status"] == "completed" else None
            items.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "name": r["name"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "artifact_link": r["artifact_link"],
                    "answer": (
                        {
                            "id": ans["id"],
                            "body": ans["body"],
                            "created_at": ans["created_at"],
                        }
                        if ans
                        else None
                    ),
                }
            )
        return {"generated_at": _inbox_ts(), "items": items, "count": len(items)}

    return web_app


def _control_secret() -> modal.Secret:
    # Only the owner allowlist, not the whole .env: control-api is public-facing
    # and needs none of the API keys. Same .env lookup as Secret.from_dotenv()
    # (cwd only). In the container the deployed secret is used, so skip the read.
    if not modal.is_local():
        return modal.Secret.from_dict({})
    from dotenv import dotenv_values, find_dotenv

    email = dotenv_values(find_dotenv(".env", usecwd=True)).get("CONTROL_OWNER_EMAIL") or ""
    return modal.Secret.from_dict({"CONTROL_OWNER_EMAIL": email})


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[_control_secret()],
    # His call, 11-09: no always-on container, and stay warm at most 2 minutes
    # after a visit. Costs pennies (~$0.40/mo at five visits a day) and buys
    # back the ~4.7 s cold start for the second and later loads of a session.
    scaledown_window=120,
)
# Without this every container answers one request at a time, so the panel's
# four parallel calls serialise — or each boot its own cold container, which
# made the round trips below invisible in practice. Read-only handlers, so 8
# concurrent inputs is safe; it is what lets one warm container serve a burst.
@modal.concurrent(max_inputs=8)
@modal.asgi_app(label="control-api")
def control_api():
    """Private control endpoint: read-only view over the agent's own files.

    This is the data side of the /control page on the gateway. Auth is
    forward-auth to the site's Netlify Identity: the SPA sends the Identity
    access JWT, and this API re-validates it against
    https://sudarshanaai.netlify.app/.netlify/identity/user before serving
    anything. (GoTrue signs tokens HS256 with a server-side secret — there is
    no public JWKS — so re-validating via the live endpoint is the only sound
    check.) The validated user's email must then match CONTROL_OWNER_EMAIL
    from .env. No write endpoints; every path is validated against a fixed
    directory allowlist and strict basename rules — no traversal.
    """
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import os as _os
    import re as _re
    import time as _time
    from datetime import datetime as _datetime, timedelta as _timedelta, timezone as _timezone
    import requests as _requests

    CONV_DIR = _os.path.join(VOLUME_PATH, "conversation_history")
    LOG_DIR = _os.path.join(VOLUME_PATH, "logs")
    ACTION_DIR = _os.path.join(VOLUME_PATH, "actions")
    MEMORY_DIR = _os.path.join(VOLUME_PATH, "memory")
    # The public site the panel lives on and which hosts the Identity backend.
    IDENTITY_BASE = "https://sudarshanaai.netlify.app"
    OWNER_EMAIL = _os.environ.get("CONTROL_OWNER_EMAIL", "").strip().lower()

    deps = make_deps()
    web_app = FastAPI(title="sudarshana-control")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://sudarshanaai.netlify.app"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["authorization", "accept"],
    )

    # Every request logs one line with its own wall time, so the next latency
    # question ("which call is slow now?") is answered by the Modal logs
    # instead of by a stopwatch in a browser. Paths and statuses only — never
    # a token, a payload or a file body. In a finally block so a 500 is timed
    # too instead of vanishing from the log. The number is in-container time:
    # it starts after the container is up, so it does NOT include the cold
    # start — never quote it as the whole load time.
    @web_app.middleware("http")
    async def _timing(request, call_next):
        _t0 = _time.monotonic()
        _status = 500
        try:
            response = await call_next(request)
            _status = response.status_code
            return response
        finally:
            _ms = int((_time.monotonic() - _t0) * 1000)
            print(f"[control-api] {request.method} {request.url.path} {_status} {_ms}ms")

    _SESSION_RE = _re.compile(r"^session_[0-9a-f]{32}\.md$")
    _DATE_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    _ACTION_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.md$")

    def _safe_join(root: str, name: str) -> str:
        # Strict basename + extension allowlist: no separators, no leading
        # dots, no traversal. root is a fixed allowlisted constant.
        if not name:
            raise HTTPException(status_code=400, detail="empty name")
        if name != _os.path.basename(name) or name in (".", "..") or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid name")
        if "/" in name or "\\" in name or "\x00" in name:
            raise HTTPException(status_code=400, detail="invalid name")
        return _os.path.join(root, name)

    # --- latency: two costs on the hot path were repeat work -----------------
    # Measured 2026-11-09 from the operator's browser: ~8-10 s for the first
    # panel load, 2-4 s for a reload right after. This function does no heavy
    # computation — it reads files — so the time went to two things done on
    # *every* request: a forward-auth round trip to Netlify Identity, and a
    # full volume.reload(). A panel load fires four requests at once, so both
    # were paid four times. (The cold start is separate and not addressable
    # from in here: 11-09 it was mostly file count on the Volume — 207K files,
    # ~97% Lean tooling — and after that was cleared it measured ~4.7 s, of
    # which ~3 s is an empty container and a slim image would not help.)
    import hashlib as _hashlib
    import threading as _threading

    # How long a token that proved it belongs to the owner is trusted in this
    # container. Short enough that a revoked session stops working quickly;
    # long enough that a panel load pays one round trip, not four.
    _AUTH_TTL = 120.0
    _AUTH_CACHE: dict[str, tuple[str, float]] = {}
    _AUTH_LOCK = _threading.Lock()
    # A panel load fires its four requests at once with the same token, so a
    # cache miss would otherwise be four forward-auth round trips in parallel.
    # One fill lock *per token*: two tabs (or a second device) must not queue
    # behind each other the way they would on a single container-wide lock.
    _FILL_LOCKS: dict = {}
    # If Identity is down, remember how long to stop asking. A container serving
    # 8 concurrent inputs must not stack one 6 s timeout per request.
    _IDENTITY_DOWN_UNTIL = {"at": 0.0}
    _IDENTITY_BREAKER = 10.0
    # A container re-syncs the Volume at most this often, so a burst of panel
    # requests waits for one reload instead of running four.
    _RELOAD_MIN_INTERVAL = 5.0
    _RELOAD_STATE = {"at": 0.0}
    _RELOAD_LOCK = _threading.Lock()
    # A reload swaps the Volume's view underneath whoever is reading. Readers
    # register while they touch files; a reload waits for them to finish and
    # holds new readers out for its duration, so a concurrent read can never
    # see a half-swapped volume (a blank file, a vanished directory).
    _VOL_COND = _threading.Condition()
    _READERS = {"n": 0}
    _RELOADING = {"on": False}

    class _VolumeRead:
        """Context manager for every read of the mounted Volume."""

        def __enter__(self):
            with _VOL_COND:
                while _RELOADING["on"]:
                    # Bounded: never let a stuck reload wedge a request.
                    if not _VOL_COND.wait(timeout=5.0):
                        break
                _READERS["n"] += 1
            return self

        def __exit__(self, *_exc):
            with _VOL_COND:
                _READERS["n"] = max(0, _READERS["n"] - 1)
                if _READERS["n"] == 0:
                    _VOL_COND.notify_all()
            return False

    def _fill_lock(key: str):
        with _AUTH_LOCK:
            lk = _FILL_LOCKS.get(key)
            if lk is None:
                if len(_FILL_LOCKS) > 64:  # one owner, a few tabs
                    _FILL_LOCKS.clear()
                lk = _FILL_LOCKS[key] = _threading.Lock()
            return lk

    def _auth_lookup(token: str):
        """(cache key, cached email or None) — never validates on a miss."""
        key = _hashlib.sha256(token.encode("utf-8")).hexdigest()
        with _AUTH_LOCK:
            hit = _AUTH_CACHE.get(key)
            if hit is not None and hit[1] > _time.time():
                return key, hit[0]
            _AUTH_CACHE.pop(key, None)
            return key, None

    def _auth_store(key: str, email: str) -> None:
        with _AUTH_LOCK:
            if len(_AUTH_CACHE) > 64:  # one owner, a few tabs; keep it bounded
                _AUTH_CACHE.clear()
            _AUTH_CACHE[key] = (email, _time.time() + _AUTH_TTL)

    def _refresh_volume() -> None:
        """Reload the Volume, at most once every few seconds, across threads.

        A warm container sees the Volume as of its start, so the panel must
        reload to see what the agent has committed since — but it does not
        need to do it once per request, and it must not swap the view out from
        under a read that is in flight.
        """
        with _RELOAD_LOCK:
            if _time.time() - _RELOAD_STATE["at"] < _RELOAD_MIN_INTERVAL:
                return
            # Park new readers and wait (briefly) for the ones already inside.
            # Monotonic, not wall clock: a clock step must not cut the wait
            # short and let a reload swap the view under a live read.
            with _VOL_COND:
                _RELOADING["on"] = True
                _deadline = _time.monotonic() + 5.0
                while _READERS["n"] > 0:
                    _left = _deadline - _time.monotonic()
                    if _left <= 0:
                        break
                    _VOL_COND.wait(timeout=_left)
            try:
                deps.storage.reload()
                _RELOAD_STATE["at"] = _time.time()
            except Exception:  # noqa: BLE001
                # No stamp on failure, so the next request retries: a stale
                # view is worse than a slow one.
                _RELOAD_STATE["at"] = 0.0
            finally:
                with _VOL_COND:
                    _RELOADING["on"] = False
                    _VOL_COND.notify_all()

    def _owner(authorization: str | None = Header(default=None, alias="Authorization")):
        if not OWNER_EMAIL:
            raise HTTPException(
                status_code=503,
                detail="CONTROL_OWNER_EMAIL not configured — set it in .env and redeploy",
            )
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if not token:
            raise HTTPException(status_code=401, detail="empty bearer token")
        key, email = _auth_lookup(token)
        if email is None:
            with _fill_lock(key):
                # Re-check: another request for this token may have just
                # validated it while we waited.
                key, email = _auth_lookup(token)
                if email is None:
                    # If Identity just failed, answer now instead of paying
                    # another 6 s timeout on top of the last one.
                    if _time.time() < _IDENTITY_DOWN_UNTIL["at"]:
                        raise HTTPException(
                            status_code=503, detail="identity service unreachable"
                        )
                    try:
                        resp = _requests.get(
                            f"{IDENTITY_BASE}/.netlify/identity/user",
                            headers={"Authorization": f"Bearer {token}"},
                            # One request per token validates; the rest of the
                            # burst reuses it. Identity answers in well under a
                            # second, so 6 s fails fast rather than stalling.
                            timeout=6,
                        )
                    except _requests.RequestException:
                        _IDENTITY_DOWN_UNTIL["at"] = _time.time() + _IDENTITY_BREAKER
                        raise HTTPException(
                            status_code=503, detail="identity service unreachable"
                        )
                    if resp.status_code >= 500:
                        # Their outage, not a bad token: trip the breaker too,
                        # or every queued request pays its own timeout.
                        _IDENTITY_DOWN_UNTIL["at"] = _time.time() + _IDENTITY_BREAKER
                        raise HTTPException(
                            status_code=503, detail="identity service unavailable"
                        )
                    if resp.status_code != 200:
                        raise HTTPException(status_code=401, detail="invalid token")
                    try:
                        user = resp.json()
                    except ValueError:
                        raise HTTPException(
                            status_code=503, detail="identity service returned invalid JSON"
                        )
                    email = (user.get("email") or "").strip().lower()
                    if email != OWNER_EMAIL:
                        raise HTTPException(
                            status_code=403, detail="authenticated, but not the control owner"
                        )
                    # Cache only a token that proved it is the owner: a bad
                    # token or someone else's re-validates (and is refused)
                    # every time.
                    _auth_store(key, email)
        # Only after auth, so anonymous requests can't drive reloads.
        _refresh_volume()
        return {"email": email}

    def _audit(user: dict, what: str) -> None:
        # Best-effort request trace in the Modal logs only — never in a payload.
        print(f"[control-api] {user.get('email')} {what} at {_time.time():.0f}")

    def _read_text(path: str) -> str | None:
        with _VolumeRead():
            if not _os.path.isfile(path):
                return None
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

    # Logs and action files ship whole (sessions are paged); cap them so one
    # runaway file can't blow up a response. Largest today is ~150 KiB.
    MAX_FILE_CHARS = 2_000_000

    def _read_capped(path: str) -> dict | None:
        with _VolumeRead():
            if not _os.path.isfile(path):
                return None
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(MAX_FILE_CHARS + 1)
            truncated = len(text) > MAX_FILE_CHARS
            return {
                "size": _os.path.getsize(path),
                "truncated": truncated,
                "content": text[:MAX_FILE_CHARS],
            }

    def _list_files(root: str) -> list:
        with _VolumeRead():
            if not _os.path.isdir(root):
                return []
            out = []
            for name in sorted(_os.listdir(root)):
                p = _os.path.join(root, name)
                if not _os.path.isfile(p):
                    continue
                st = _os.stat(p)
                out.append(
                    {
                        "id": name,
                        "size": st.st_size,
                        "modified": _datetime.fromtimestamp(st.st_mtime, _timezone.utc).isoformat(),
                    }
                )
            return out

    @web_app.get("/api/control/sessions")
    def list_sessions(user: dict = Depends(_owner)):
        _audit(user, "list_sessions")
        return {"sessions": _list_files(CONV_DIR)}

    @web_app.get("/api/control/sessions/{session_id}")
    def read_session(
        session_id: str,
        user: dict = Depends(_owner),
        page: int = 1,
        per_page: int = 200,
        tail: int = 0,
    ):
        # Session transcripts grow every turn (the live one is ~8 MB); always
        # slice server-side and never ship more than a bounded window.
        if not _SESSION_RE.match(session_id):
            raise HTTPException(status_code=400, detail="invalid session id")
        path = _safe_join(CONV_DIR, session_id)
        with _VolumeRead():
            if not _os.path.isfile(path):
                raise HTTPException(status_code=404, detail="no such session")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()  # reading in memory is fine; only a window ships
        total = len(lines)
        if tail > 0:
            n = min(int(tail), 2000, total)
            body = lines[-n:]
            _audit(user, f"read_session {session_id} tail={n}")
            return {
                "id": session_id,
                "total_lines": total,
                "start_line": total - n,
                "end_line": total,
                "lines": body,
            }
        per = max(1, min(int(per_page), 500))
        pg = max(1, int(page))
        start = (pg - 1) * per
        _audit(user, f"read_session {session_id} page={pg}")
        return {
            "id": session_id,
            "total_lines": total,
            "page": pg,
            "per_page": per,
            "start_line": start,
            "end_line": min(total, start + per),
            "lines": lines[start : start + per],
        }

    @web_app.get("/api/control/logs")
    def list_logs(user: dict = Depends(_owner)):
        _audit(user, "list_logs")
        return {"logs": _list_files(LOG_DIR)}

    @web_app.get("/api/control/logs/{log_date}")
    def read_log(log_date: str, user: dict = Depends(_owner)):
        if not _DATE_RE.match(log_date):
            raise HTTPException(status_code=400, detail="invalid log date")
        path = _safe_join(LOG_DIR, f"{log_date}.md")
        doc = _read_capped(path)
        if doc is None:
            raise HTTPException(status_code=404, detail="no log for that date")
        _audit(user, f"read_log {log_date}")
        return {"date": log_date, **doc}

    @web_app.get("/api/control/state")
    def read_state(user: dict = Depends(_owner)):
        _audit(user, "read_state")
        files = {
            "state_md": _os.path.join(MEMORY_DIR, "state.md"),
            "decisions_md": _os.path.join(MEMORY_DIR, "decisions.md"),
            "roadmap_md": _os.path.join(VOLUME_PATH, "ROADMAP.md"),
        }
        out, missing = {}, []
        for key, path in files.items():
            text = _read_text(path)
            if text is None:
                missing.append(key)
            else:
                out[key] = text
        return {"files": out, "missing": missing}

    @web_app.get("/api/control/actions")
    def list_actions(user: dict = Depends(_owner)):
        _audit(user, "list_actions")
        return {"actions": _list_files(ACTION_DIR)}

    @web_app.get("/api/control/actions/{action_id}")
    def read_action(action_id: str, user: dict = Depends(_owner)):
        if not _ACTION_RE.match(action_id):
            raise HTTPException(status_code=400, detail="invalid action id")
        path = _safe_join(ACTION_DIR, action_id)
        doc = _read_capped(path)
        if doc is None:
            raise HTTPException(status_code=404, detail="no such action file")
        _audit(user, f"read_action {action_id}")
        return {"id": action_id, **doc}

    # --- roadmap: initiatives -> tasks, from the store the roadmap tools own --
    # Served as records, not prose: the panel renders status, tasks and the
    # doc link without parsing anything. `doc_exists` is checked here so a
    # doc deleted from the shell shows as missing, not as a dead link.
    def _roadmap_payload(it: dict) -> dict:
        with _VolumeRead():
            doc_exists = _os.path.isfile(it.get("doc") or "")
        tasks = it.get("tasks") or []
        return {
            **{k: v for k, v in it.items() if k not in ("tasks",)},
            "doc_exists": doc_exists,
            "tasks": tasks,
            "tasks_open": sum(1 for t in tasks if t.get("status") in TASK_OPEN),
            "tasks_done": sum(1 for t in tasks if t.get("status") == "done"),
        }

    def _roadmap_read():
        """(initiatives in roadmap order, error or None)."""
        try:
            with _VolumeRead():
                items = _roadmap_load()
        except Exception as e:  # noqa: BLE001
            return [], f"can't read the roadmap store: {type(e).__name__}"
        return _rm_sorted(items), None

    @web_app.get("/api/control/roadmap")
    def read_roadmap(user: dict = Depends(_owner)):
        _audit(user, "read_roadmap")
        items, err = _roadmap_read()
        focus = _roadmap_focus(items) if items else None
        return {
            "initiatives": [_roadmap_payload(it) for it in items],
            # The initiative the next cycle works on, by the same rule the
            # agent's per-turn note uses.
            "focus": focus["id"] if focus else None,
            # None when readable; a short reason otherwise, so an empty
            # roadmap never passes for "the store is broken".
            "error": err,
        }

    @web_app.get("/api/control/roadmap/{initiative_id}")
    def read_initiative(initiative_id: str, user: dict = Depends(_owner)):
        if not _re.fullmatch(ROADMAP_ID_RE, initiative_id or ""):
            raise HTTPException(status_code=400, detail="invalid initiative id")
        items, err = _roadmap_read()
        if err:
            raise HTTPException(status_code=503, detail=err)
        it = _rm_find(items, initiative_id)
        if not it:
            raise HTTPException(status_code=404, detail="no such initiative")
        _audit(user, f"read_initiative {initiative_id}")
        return _roadmap_payload(it)

    @web_app.get("/api/control/roadmap/{initiative_id}/doc")
    def read_initiative_doc(initiative_id: str, user: dict = Depends(_owner)):
        if not _re.fullmatch(ROADMAP_ID_RE, initiative_id or ""):
            raise HTTPException(status_code=400, detail="invalid initiative id")
        items, err = _roadmap_read()
        if err:
            raise HTTPException(status_code=503, detail=err)
        it = _rm_find(items, initiative_id)
        if not it:
            raise HTTPException(status_code=404, detail="no such initiative")
        # The doc path comes from the store, so confine it to the two
        # directories the tools write docs to before reading it.
        path = _os.path.realpath(it.get("doc") or "")
        if _os.path.dirname(path) not in (_os.path.realpath(ACTION_DIR), _os.path.realpath(ACTIONS_ARCHIVE_DIR)):
            raise HTTPException(status_code=400, detail="doc path outside the actions directory")
        doc = _read_capped(path)
        if doc is None:
            raise HTTPException(status_code=404, detail="doc missing")
        _audit(user, f"read_initiative_doc {initiative_id}")
        return {"id": initiative_id, "doc": it["doc"], **doc}

    # --- overview: the three questions the /control page actually asks ------
    # Everything above serves *files*; a person reading the panel has three
    # questions instead — am I OK, what do you need from me, what have you
    # done. Answering them in the front end would mean parsing my own notes,
    # which is a fake answer. So it is assembled here, from real sources:
    #   health      -> the status Dict (turn:* records + `current`)
    #   asks        -> memory/asks.json, open asks, written by ask_rinkesh / resolve_ask
    #   recent_work -> the `## ` headings of the last two daily logs
    ASKS_FILE = ASKS_PATH  # written by the ask tools; one home for the path
    # The operator reads the panel in Pacific time; "today" must mean his day.
    # ZoneInfo carries the real rules (tzdata is in the image — _timestamp has
    # used it since the beginning), so this no longer hand-rolls the US DST
    # calendar and cannot drift from what _timestamp reports.
    from zoneinfo import ZoneInfo as _ZoneInfo

    PACIFIC = _ZoneInfo("America/Los_Angeles")

    def _local_date(iso) -> str:
        ts = _status_parse_ts(iso)
        if ts is None:
            return ""
        return _datetime.fromtimestamp(ts, PACIFIC).date().isoformat()

    def _turn_records():
        """(turn records, `current`, error) from ONE read of the status Dict.
        `current` lives in the same Dict, so it comes from this fetch too: a
        second get() was an extra round trip, and could describe a different
        moment than the records it sat beside.

        Returning a bare `[]` on a Dict read failure made "I can see no turns"
        indistinguishable from "I cannot see my own records": the panel would
        show 0 turns today and 0 errors — a quiet, healthy-looking agent — when
        the truth was that it had gone blind. Same lie-mode as asks_error and
        recent_work_error, in the section that answers "am I OK".
        """
        try:
            # One fetch for the whole Dict. A get() per turn key meant ~100
            # sequential round-trips per panel load; items() is one call, and
            # it fails as a unit rather than half-reading.
            items = list(deps.status_kv.items() or [])
        except Exception:  # noqa: BLE001
            return None, {}, "the status store didn't answer"
        recs = []
        cur = {}
        for k, rec in items:
            if k == "current" and isinstance(rec, dict):
                cur = rec
                continue
            if not (isinstance(k, str) and k.startswith("turn:")):
                continue
            if isinstance(rec, dict) and rec.get("started"):
                recs.append(rec)
        recs.sort(key=lambda r: str(r.get("started")))
        return recs, cur, None

    def _asks():
        """Return (asks, error). An unreadable or malformed asks.json is NOT an
        empty list: "no asks" and "the list is broken" look identical to a
        reader, and the panel would then claim I need nothing from him. The
        caller surfaces the error so the panel can say it can't read the list.
        """
        import json as _json

        try:
            raw = _read_text(ASKS_FILE)
        except Exception:  # noqa: BLE001
            # The file exists but the read failed. Left uncaught this raised out
            # of the handler and 500'd the whole panel — losing health and the
            # timeline too, over one bad file.
            return None, "couldn't read asks.json"
        if not raw:
            return [], None
        try:
            data = _json.loads(raw)
        except ValueError:
            return None, "asks.json is not valid JSON"
        items = data.get("asks") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return None, "asks.json has no asks list"
        now = _time.time()
        out = []
        for it in items[:20]:
            if not isinstance(it, dict):
                continue
            asked_at = str(it.get("asked_at") or "").strip()
            since = _status_parse_ts(asked_at)
            out.append(
                {
                    "id": str(it.get("id") or "")[:64],
                    "ask": str(it.get("ask") or "")[:300],
                    "detail": str(it.get("detail") or "")[:600],
                    "asked_at": asked_at,
                    "waiting_hours": (
                        round((now - since) / 3600.0, 1) if since is not None else None
                    ),
                }
            )
        return out, None

    def _recent_work(limit: int = 12):
        """Return (items, error). Same rule as _asks(): an unreadable log must
        not come back as [] — the panel would show an empty timeline for a day
        I did plenty, which reads as "nothing happened". Readable logs that
        simply have no entries yet are [] with no error, which is the truth.
        """
        try:
            names = sorted(
                n
                for n in _os.listdir(LOG_DIR)
                if n.endswith(".md") and _DATE_RE.match(n[:-3])
            )[-2:]
        except OSError:
            return None, "can't read my log directory"
        out = []
        err = None
        for name in names:
            try:
                text = _read_text(_os.path.join(LOG_DIR, name))
            except Exception:  # noqa: BLE001
                text = None
            if text is None:
                err = err or f"couldn't read {name}"
                continue
            for line in text.splitlines():
                if not line.startswith("## "):
                    continue
                head = line[3:].strip()
                stamp, _, body = head.partition("—")
                body = body.strip() or head
                tokens = stamp.strip().split(" ")
                out.append(
                    {
                        "date": name[:-3],
                        "time": tokens[0] if tokens and tokens[0] else "",
                        "text": body[:240],
                    }
                )
        if out:
            return out[-limit:], err
        # Nothing to show splits in two, and the panel must not conflate them:
        # the logs read fine but have no `## ` entries yet (just after midnight)
        # is an honest empty — [] with no reason. Only a read that failed and
        # produced nothing is None, because an empty list there would render as
        # "nothing happened" over a day I actually worked.
        return ([] if err is None else None), err

    def _count_errors(recs: list, now_ts: float) -> int:
        # A turn is "wrong" if it died mid-run (unclosed past the zombie TTL)
        # or closed on an outcome other than done. A turn still inside the TTL
        # is simply running — counting it would make the verdict say "not OK"
        # exactly when he looks mid-cycle, which is most of the time.
        # Older than 24h is history, not health.
        errors = 0
        for r in recs:
            ts = _status_parse_ts(r.get("started"))
            if ts is None or (now_ts - ts) > 86400:
                continue
            outcome = str(r.get("outcome") or "")
            if r.get("ended") is None:
                if _status_is_zombie(r):
                    errors += 1
                continue
            if outcome != "done":
                errors += 1
        return errors

    def _overview_report(recs) -> dict:
        """The latest report with its progress filled in from the roadmap.
        A report from before worked_on (it typed project/done/left itself)
        keeps its own fields: setdefault never overwrites them."""
        rep = _report_latest(recs)
        if rep.get("run") is None:
            return rep
        items, err = _roadmap_read()
        rep = _report_with_progress(rep, items)
        rep["roadmap_error"] = err
        return rep

    @web_app.get("/api/control/overview")
    def overview(user: dict = Depends(_owner)):
        _audit(user, "overview")
        recs, cur, records_err = _turn_records()
        recs = recs or []
        now_ts = _time.time()
        last = recs[-1] if recs else None
        today = _local_date(_status_ts())
        errors = _count_errors(recs, now_ts)
        asks_ok, asks_err = _asks()
        work_ok, work_err = _recent_work()
        return {
            # `ok` = this request succeeded. `health.ok` = the verdict about me.
            "ok": True,
            "generated_at": _status_ts(),
            "health": {
                "ok": bool(recs) and errors == 0,
                "state": str(cur.get("state") or "unknown"),
                "since": cur.get("since"),
                "last_cycle": (last or {}).get("started"),
                "last_outcome": (last or {}).get("outcome"),
                "turns_today": sum(
                    1 for r in recs if _local_date(r.get("started")) == today
                ),
                "errors_24h": errors,
                # None when the status Dict was readable; a short reason
                # otherwise, so "no turns today" cannot stand in for "I went
                # blind". The panel drops the numbers when this is set rather
                # than rendering zeros it did not measure.
                "records_error": records_err,
            },
            "asks": asks_ok,
            # None when the list is readable; a short human reason otherwise.
            # The panel shows "can't read the ask list" rather than "nothing".
            "asks_error": asks_err,
            "recent_work": work_ok,
            # None when my logs are readable; a short reason otherwise, so an
            # empty timeline never passes for "nothing happened".
            "recent_work_error": work_err,
            # What I'm doing, why, and where the work stands: the latest
            # report_status, read from the same turn records as `health`.
            # {doing, why, project, done_when, done, left, next, reported_at,
            # run, fresh, missed_runs}; fresh=False when finished runs since
            # didn't report — the panel shows it as out of date rather than as
            # current. run=None when no run has reported. None when the records
            # couldn't be read.
            "report": None if records_err else _overview_report(recs),
        }

    return web_app


@app.function(
    image=image,
    # spawn() returns once the call is queued, so a short timeout is enough.
    timeout=60,
    schedule=modal.Cron("0 * * * *"),
)
def hourly_trigger():
    # Bare cron wrapper — schedule= isn't allowed on @modal.method().
    # spawn(), not remote(): a blocking wrapper sat for the whole run, and
    # when Modal preempted it the retry started a second, overlapping run.
    Sudarshana().hourly_checkin.spawn()
