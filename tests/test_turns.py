"""The steps around run_turn: telegram_turn and hourly_turn.

Runs the real turn functions on fake deps with run_turn stubbed out. No
model, no network, no Modal.

Why: these steps used to live inside the Modal class, where no test could
reach them. Any runtime now calls these, so they pin what every turn must do:
a resent Telegram update never runs twice, /agents never reaches the model,
a turn is recorded as started before the model runs and closed with its real
outcome, the reply goes back to Telegram (or, for the hourly turn, to the
owner), and storage is committed at the end.
"""
import pytest

from conftest import FakeStorage, fake_deps
from sudarshana.core import turns
from sudarshana.core.agent import STEP_LIMIT, TurnResult
from sudarshana.infra.storage.local.stores import DictKV
from sudarshana.interfaces.telegram import turn as tg_turn


class _Agent:
    a2a = object()


@pytest.fixture()
def env(monkeypatch):
    deps = fake_deps(lambda name: DictKV(), storage=FakeStorage())
    events = []
    sent = []
    outcome = {"value": "done"}

    def fake_run_turn(agent, message, trigger, turn_key=None):
        events.append(("run_turn", message, trigger, turn_key))
        return TurnResult(reply=f"reply to {message}", outcome=outcome["value"])

    class _Telegram:
        def send(self, text):
            sent.append(text)

    monkeypatch.setattr(tg_turn, "run_turn", fake_run_turn)
    monkeypatch.setattr(turns, "run_turn", fake_run_turn)
    monkeypatch.setattr(tg_turn, "TelegramNotifier", _Telegram)
    monkeypatch.setattr(tg_turn, "_agents_command", lambda a2a, text: f"agents: {text}")
    monkeypatch.setattr(turns, "_a2a_keep_alive", lambda store: events.append(("keep_alive",)))
    return deps, events, sent, outcome


def _record(deps, key):
    return deps.status_kv.get(key)


def test_a_telegram_message_runs_one_recorded_turn(env):
    deps, events, sent, _ = env
    tg_turn.telegram_turn(_Agent(), deps, "hello", 7)
    [(_, message, trigger, key)] = events
    assert (message, trigger) == ("hello", "telegram")
    rec = _record(deps, key)
    assert rec["state"] == "running-telegram" and rec["outcome"] == "done" and rec["ended"]
    assert sent == ["reply to hello"]
    assert deps.storage.commits == 1


def test_a_resent_update_does_not_run_twice(env):
    deps, events, sent, _ = env
    tg_turn.telegram_turn(_Agent(), deps, "hello", 7)
    tg_turn.telegram_turn(_Agent(), deps, "hello", 7)
    assert len(events) == 1 and sent == ["reply to hello"]


def test_agents_commands_never_reach_the_model(env):
    deps, events, sent, _ = env
    tg_turn.telegram_turn(_Agent(), deps, "/agents list", 8)
    assert events == [] and sent == ["agents: /agents list"]
    assert deps.storage.commits == 0


def test_a_step_limit_is_recorded_as_such(env):
    deps, events, sent, outcome = env
    outcome["value"] = STEP_LIMIT
    tg_turn.telegram_turn(_Agent(), deps, "big job", None)
    [(_, _, _, key)] = events
    assert _record(deps, key)["outcome"] == STEP_LIMIT


def test_the_hourly_turn_reports_to_the_owner(env):
    deps, events, sent, _ = env
    turns.hourly_turn(_Agent(), deps)
    assert events[0] == ("keep_alive",)
    (_, message, trigger, key) = events[1]
    assert trigger == "hourly" and message == turns.HOURLY_TASK
    assert _record(deps, key)["state"] == "running-hourly"
    assert deps.owner_notifier.sent == [f"reply to {turns.HOURLY_TASK}"]
    assert sent == []  # not the Telegram reply path
    assert deps.storage.commits == 1
