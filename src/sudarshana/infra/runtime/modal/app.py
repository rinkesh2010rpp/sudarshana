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
from sudarshana.interfaces.http.control import create_control_app
from sudarshana.interfaces.http.inbox import create_inbox_app
from sudarshana.interfaces.http.status import create_status_app
from sudarshana.interfaces.telegram.send import TelegramNotifier
from sudarshana.interfaces.telegram.webhook import (
    TELEGRAM_CLAIM_DICT_NAME,
    _telegram_record,
    _telegram_seen,
    handle_update,
)
from sudarshana.memory.status import STATUS_DICT_NAME, _status_turn_end, _status_turn_start

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
    """status-center tier 1: read-only /api/status + /api/events. The app is interfaces/http/status.py."""
    return create_status_app(make_deps())


@app.function(image=image, volumes={VOLUME_PATH: volume}, secrets=[modal.Secret.from_dotenv()])
@modal.asgi_app(label="inbox-api")
def inbox_api():
    """Visitor inbox endpoint. The app is interfaces/http/inbox.py."""
    return create_inbox_app(make_deps())


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
    """Private control endpoint: read-only view over the agent's own files. The app is interfaces/http/control.py."""
    return create_control_app(make_deps())


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
