"""The ask list: ask_rinkesh / resolve_ask and the per-turn note.

Runs the real `agent/main.py` helpers against an in-memory stand-in for
`modal.Dict` and a recording stand-in for the Telegram send. No network.

Why: what the agent needed from Rinkesh lived as a hand-copied "his:" line in
state.md and went stale (PR #64 was still listed days after it merged). These
tests pin the properties that fix that: an ask is recorded only once he has
been sent it, it closes by an explicit outcome or by lapsing, and a lapse is
reported once rather than every hour.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import modal  # noqa: E402

STORE = {}
T0 = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


class _FakeDict:
    def __init__(self, name):
        self.name = name

    def get(self, key, default=None):
        return STORE.get((self.name, key), default)

    def put(self, key, value):
        STORE[(self.name, key)] = value


@pytest.fixture()
def mod(monkeypatch):
    STORE.clear()
    monkeypatch.setattr(
        modal.Dict, "from_name", classmethod(lambda cls, name, create_if_missing=False: _FakeDict(name))
    )
    sys.modules.pop("main", None)
    import main

    return main


@pytest.fixture()
def sent():
    return []


def _ask(mod, sent, q="Merge PR #70?", now=T0, **kw):
    kw.setdefault("why", "it's blocking the deploy")
    return mod._ask_create(q, now=now, send=sent.append, **kw)


def test_ask_sends_one_message_and_records_it(mod, sent):
    res = _ask(mod, sent, options="A (recommended) / B")
    assert res["ok"]
    assert len(sent) == 1
    msg = sent[0]
    assert f"[ask {res['id']}]" in msg and "Merge PR #70?" in msg
    assert "Why: it's blocking the deploy" in msg and "Options: A (recommended) / B" in msg
    [a] = mod._asks_load()
    assert a["status"] == "open" and a["id"] == res["id"]
    assert mod._parse_ts(a["expires"]) == T0 + timedelta(days=3)


def test_failed_send_records_nothing(mod):
    def boom(_):
        raise RuntimeError("telegram down")

    res = mod._ask_create("Q?", "why", now=T0, send=boom)
    assert not res["ok"] and "not recorded" in res["error"]
    assert mod._asks_load() == []


@pytest.mark.parametrize("q,why", [("", "why"), ("Q?", ""), ("  ", "  ")])
def test_question_and_why_are_required(mod, sent, q, why):
    assert not mod._ask_create(q, why, now=T0, send=sent.append)["ok"]
    assert sent == []


def test_duplicate_open_question_is_refused(mod, sent):
    first = _ask(mod, sent)
    again = _ask(mod, sent, q="  merge pr #70?  ")
    assert not again["ok"] and first["id"] in again["error"]
    assert len(sent) == 1


def test_open_cap(mod, sent):
    for i in range(mod.ASK_MAX_OPEN):
        assert _ask(mod, sent, q=f"Q{i}?")["ok"]
    over = _ask(mod, sent, q="one more?")
    assert not over["ok"] and "cap" in over["error"]
    assert len(sent) == mod.ASK_MAX_OPEN
    # Closing one frees a slot.
    first = mod._asks_load()[0]["id"]
    assert mod._ask_resolve(first, "he said yes", now=T0)["ok"]
    assert _ask(mod, sent, q="one more?")["ok"]


def test_expiry_days_are_clamped(mod, sent):
    a = _ask(mod, sent, q="long?", expires_days=99)
    b = _ask(mod, sent, q="short?", expires_days=0)
    assert mod._parse_ts(a["expires"]) == T0 + timedelta(days=mod.ASK_MAX_DAYS)
    assert mod._parse_ts(b["expires"]) == T0 + timedelta(days=1)
    assert not _ask(mod, sent, q="bad?", expires_days="soon")["ok"]


def test_resolve(mod, sent):
    ask_id = _ask(mod, sent)["id"]
    assert not mod._ask_resolve(ask_id, "", now=T0)["ok"]
    assert not mod._ask_resolve(ask_id, "x", status="done", now=T0)["ok"]
    assert not mod._ask_resolve("ffff", "x", now=T0)["ok"]
    # The id as it appears in the message is accepted too.
    assert mod._ask_resolve(f"[ask {ask_id}]", "merged already", status="dropped", now=T0)["ok"]
    [a] = mod._asks_load()
    assert (a["status"], a["outcome"]) == ("dropped", "merged already")
    again = mod._ask_resolve(ask_id, "x", now=T0)
    assert not again["ok"] and "already dropped" in again["error"]


def test_context_empty_when_nothing_open(mod):
    assert mod._asks_context(now=T0) == ""


def test_context_lists_open_asks(mod, sent):
    ask_id = _ask(mod, sent)["id"]
    ctx = mod._asks_context(now=T0 + timedelta(hours=5))
    assert f"[{ask_id}] asked 5h ago, lapses in 2d: Merge PR #70?" in ctx
    assert "resolve_ask" in ctx


def test_lapse_is_reported_once_then_silent(mod, sent):
    ask_id = _ask(mod, sent)["id"]
    later = T0 + timedelta(days=3, minutes=1)
    ctx = mod._asks_context(now=later)
    assert "lapsed" in ctx and f"[{ask_id}]" in ctx and "Open asks" not in ctx
    assert mod._asks_context(now=later + timedelta(hours=1)) == ""
    [a] = mod._asks_load()
    assert a["status"] == "expired"
    # A lapsed ask can't be resolved, and asking the same question again is allowed.
    assert not mod._ask_resolve(ask_id, "x", now=later)["ok"]
    assert _ask(mod, sent, now=later)["ok"]


def test_closed_history_is_bounded(mod, sent):
    for i in range(mod.ASK_KEEP_CLOSED + 5):
        ask_id = _ask(mod, sent, q=f"Q{i}?")["id"]
        mod._ask_resolve(ask_id, "ok", now=T0)
    _ask(mod, sent, q="still open?")
    asks = mod._asks_load()
    assert len(asks) == mod.ASK_KEEP_CLOSED + 1
    assert asks[0]["status"] == "open"


def test_context_fails_open(mod, monkeypatch):
    def broken():
        raise RuntimeError("dict unavailable")

    monkeypatch.setattr(mod, "_asks_store", broken)
    assert "could not be read" in mod._asks_context(now=T0)
