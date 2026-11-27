"""One turn for a Telegram message: the steps around run_turn.

Any runtime calls this once per message, one message at a time (on Modal,
the single Sudarshana container guarantees that), so the "run" dedup gate
below is the real guarantee against a message running twice.
"""
import time

from sudarshana.core.agent import run_turn
from sudarshana.integrations.a2a import _agents_command
from sudarshana.interfaces.telegram.send import TelegramNotifier
from sudarshana.interfaces.telegram.webhook import _telegram_record, _telegram_seen
from sudarshana.memory.status import _status_turn_end, _status_turn_start


def telegram_turn(agent, deps, text: str, update_id: int | None = None) -> None:
    # Second dedup gate (the webhook holds the first). Turns run one at a
    # time, so two copies of one update cannot both pass here — not even the
    # pair that races past the webhook's claim. That serialisation is why
    # this gate is the real guarantee: read-then-write here cannot interleave
    # with another copy.
    if update_id is not None:
        if _telegram_seen(deps.claims_kv, update_id, "run"):
            print(f"[telegram] duplicate update_id {update_id} — no turn started")
            return
        # Recorded before the turn so a crash mid-turn can't re-run it.
        # Both helpers swallow Dict errors: this turn was already acked to
        # Telegram, so raising here would discard the message silently.
        _telegram_record(deps.claims_kv, update_id, "run")

    # Rinkesh's external-agent controls run here, never through the model.
    if text.strip().lower().split()[:1] == ["/agents"]:
        TelegramNotifier().send(_agents_command(agent.a2a, text))
        return

    started = time.monotonic()
    print(f"[timing] process_message started: {text[:200]!r}")

    # Before any model work, so a turn that dies mid-run leaves a trace.
    turn_key = _status_turn_start(deps, "running-telegram")

    result = run_turn(agent, text, "telegram", turn_key)
    # The message came in on Telegram, so the reply goes back there.
    TelegramNotifier().send(result.reply)

    print(f"[timing] process_message finished in {time.monotonic() - started:.1f}s")
    _status_turn_end(deps, turn_key, result.outcome, "telegram turn finished")

    # Commit explicitly — the runtime may stop before any background flush
    # catches these writes.
    deps.storage.commit()
