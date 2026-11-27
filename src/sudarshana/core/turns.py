"""The scheduled turn: the steps around run_turn for the hourly wake-up.

A Telegram message's turn is interfaces/telegram/turn.py. Any runtime calls
these; the runtime only decides when.
"""
import time

from sudarshana.core.agent import run_turn
from sudarshana.core.prompt import HOURLY_TASK
from sudarshana.integrations.a2a import _a2a_keep_alive
from sudarshana.memory.status import _status_turn_end, _status_turn_start


def hourly_turn(agent, deps) -> None:
    started = time.monotonic()
    print("[timing] hourly_checkin started")

    turn_key = _status_turn_start(deps, "running-hourly")
    _a2a_keep_alive(deps.a2a_kv)
    result = run_turn(agent, HOURLY_TASK, "hourly", turn_key)
    # Nobody sent this turn a message; its report goes to the owner.
    deps.owner_notifier.send(result.reply)

    print(f"[timing] hourly_checkin finished in {time.monotonic() - started:.1f}s")
    _status_turn_end(deps, turn_key, result.outcome, "hourly turn finished")
    deps.storage.commit()
