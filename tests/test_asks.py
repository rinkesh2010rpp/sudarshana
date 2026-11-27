"""The ask list: ask_rinkesh / resolve_ask and the per-turn note.

Runs the real `sudarshana.memory.asks` helpers against a temp asks.json and a
recording stand-in for the Telegram send. No network, no Modal.

Why: what the agent needed from Rinkesh lived in hand-kept lists and went
stale (PR #64 in state.md's "his:" line, PR #61 in asks.json, each listed days
after it merged). These tests pin the properties that fix that: an ask is
recorded only once he has been sent it, and committed to the Volume at once
(not at turn end, which a dying turn never reaches); it closes by an explicit
outcome or by lapsing; and a lapse is reported once rather than every hour.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest


from conftest import FakeStorage, fake_deps  # noqa: E402

STORE = {}
COMMITS = []
DEPS = [None]  # this test's fake Deps, set by the fixture
T0 = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


class _FakeDict:
    def __init__(self, name):
        self.name = name

    def get(self, key, default=None):
        return STORE.get((self.name, key), default)

    def put(self, key, value):
        STORE[(self.name, key)] = value


@pytest.fixture()
def mod(monkeypatch, patch_all, tmp_path):
    STORE.clear()
    from sudarshana.memory import asks

    patch_all("ASKS_PATH", str(tmp_path / "memory" / "asks.json"))
    COMMITS.clear()

    def _commit():
        # Record what a commit would persist: the file as it is right now.
        COMMITS.append(Path(asks.ASKS_PATH).read_text(encoding="utf-8"))

    DEPS[0] = fake_deps(_FakeDict, storage=FakeStorage(on_commit=_commit))
    return asks


def _file(mod):
    with open(mod.ASKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _write(mod, data):
    Path(mod.ASKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(mod.ASKS_PATH).write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


@pytest.fixture()
def sent():
    return []


def _create(mod, *a, send, **kw):
    """_ask_create with `send` as the notifier."""
    DEPS[0].owner_notifier.send = send
    return mod._ask_create(DEPS[0], *a, **kw)


def _ask(mod, sent, q="Merge PR #70?", now=T0, **kw):
    kw.setdefault("why", "it's blocking the deploy")
    return _create(mod, q, now=now, send=sent.append, **kw)


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

    res = _create(mod, "Q?", "why", now=T0, send=boom)
    assert not res["ok"] and "not recorded" in res["error"]
    assert mod._asks_load() == []


@pytest.mark.parametrize("q,why", [("", "why"), ("Q?", ""), ("  ", "  ")])
def test_question_and_why_are_required(mod, sent, q, why):
    assert not _create(mod, q, why, now=T0, send=sent.append)["ok"]
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
    assert mod._ask_resolve(DEPS[0], first, "he said yes", now=T0)["ok"]
    assert _ask(mod, sent, q="one more?")["ok"]


def test_expiry_days_are_clamped(mod, sent):
    a = _ask(mod, sent, q="long?", expires_days=99)
    b = _ask(mod, sent, q="short?", expires_days=0)
    assert mod._parse_ts(a["expires"]) == T0 + timedelta(days=mod.ASK_MAX_DAYS)
    assert mod._parse_ts(b["expires"]) == T0 + timedelta(days=1)
    assert not _ask(mod, sent, q="bad?", expires_days="soon")["ok"]


def test_resolve(mod, sent):
    ask_id = _ask(mod, sent)["id"]
    assert not mod._ask_resolve(DEPS[0], ask_id, "", now=T0)["ok"]
    assert not mod._ask_resolve(DEPS[0], ask_id, "x", status="done", now=T0)["ok"]
    assert not mod._ask_resolve(DEPS[0], "ffff", "x", now=T0)["ok"]
    # The id as it appears in the message is accepted too.
    assert mod._ask_resolve(DEPS[0], f"[ask {ask_id}]", "merged already", status="dropped", now=T0)["ok"]
    [a] = mod._asks_load()
    assert (a["status"], a["outcome"]) == ("dropped", "merged already")
    again = mod._ask_resolve(DEPS[0], ask_id, "x", now=T0)
    assert not again["ok"] and "already dropped" in again["error"]


def test_context_empty_when_nothing_open(mod):
    assert mod._asks_context(DEPS[0], now=T0) == ""


def test_context_lists_open_asks(mod, sent):
    ask_id = _ask(mod, sent)["id"]
    ctx = mod._asks_context(DEPS[0], now=T0 + timedelta(hours=5))
    assert f"[{ask_id}] asked 5h ago, lapses in 2d: Merge PR #70?" in ctx
    assert "resolve_ask" in ctx


def test_lapse_is_reported_once_then_silent(mod, sent):
    ask_id = _ask(mod, sent)["id"]
    later = T0 + timedelta(days=3, minutes=1)
    ctx = mod._asks_context(DEPS[0], now=later)
    assert "lapsed" in ctx and f"[{ask_id}]" in ctx and "Open asks" not in ctx
    assert mod._asks_context(DEPS[0], now=later + timedelta(hours=1)) == ""
    [a] = mod._asks_load()
    assert a["status"] == "expired"
    # A lapsed ask can't be resolved, and asking the same question again is allowed.
    assert not mod._ask_resolve(DEPS[0], ask_id, "x", now=later)["ok"]
    assert _ask(mod, sent, now=later)["ok"]


def test_closed_history_is_bounded(mod, sent):
    for i in range(mod.ASK_KEEP_CLOSED + 5):
        ask_id = _ask(mod, sent, q=f"Q{i}?")["id"]
        mod._ask_resolve(DEPS[0], ask_id, "ok", now=T0)
    _ask(mod, sent, q="still open?")
    asks = mod._asks_load()
    assert len(asks) == mod.ASK_KEEP_CLOSED + 1
    assert asks[0]["status"] == "open"


def test_context_fails_open(mod):
    _write(mod, "{not json")
    assert "could not be read" in mod._asks_context(DEPS[0], now=T0)


def test_malformed_file_blocks_sending(mod, sent):
    _write(mod, "{not json")
    res = _ask(mod, sent)
    assert not res["ok"] and "nothing sent" in res["error"]
    assert sent == []
    assert not mod._ask_resolve(DEPS[0], "abcd", "x", now=T0)["ok"]


def test_file_has_what_control_reads(mod, sent):
    """/api/control/overview serves data["asks"] as-is and reads id, ask,
    detail, asked_at from each — so open asks only, in those fields."""
    keep = _ask(mod, sent, q="Keep?", options="A / B")["id"]
    gone = _ask(mod, sent, q="Gone?")["id"]
    mod._ask_resolve(DEPS[0], gone, "done another way", status="dropped", now=T0)
    data = _file(mod)
    assert [a["id"] for a in data["asks"]] == [keep]
    [a] = data["asks"]
    assert a["ask"] == "Keep?" and a["asked_at"] == T0.isoformat()
    assert a["detail"] == "it's blocking the deploy Options: A / B"
    assert [c["id"] for c in data["closed"]] == [gone]
    assert "do not edit by hand" in data["_what"]


def test_hand_written_asks_are_adopted_and_lapse(mod, sent):
    """The file as the agent kept it before the tools: no status, no expiry."""
    _write(mod, {
        "_what": "The agent maintains this each cycle.",
        "asks": [
            {"id": "overview-latency", "ask": "Merge + deploy PR #61", "detail": "long story",
             "asked_at": "2026-11-10T01:00:00-08:00"},
            {"id": "fresh", "ask": "Pick a colour", "detail": "",
             "asked_at": (T0 - timedelta(hours=2)).isoformat()},
            {"ask": "no timestamp"},
            "not a dict",
        ],
    })
    ctx = mod._asks_context(DEPS[0], now=T0)
    assert "[overview-latency] Merge + deploy PR #61" in ctx and "lapsed" in ctx
    assert "[fresh] asked 2h ago" in ctx
    data = _file(mod)
    assert [a["id"] for a in data["asks"]] == ["fresh"]
    assert data["closed"][0]["id"] == "overview-latency"
    assert data["closed"][0]["detail"] == "long story"
    assert mod._ask_resolve(DEPS[0], "fresh", "blue", now=T0)["ok"]


def test_every_write_is_committed_at_once(mod, sent):
    """A turn that dies after ask_rinkesh never reaches its turn-end commit."""
    ask_id = _ask(mod, sent)["id"]
    assert len(COMMITS) == 1 and ask_id in COMMITS[0]
    mod._ask_resolve(DEPS[0], ask_id, "yes", now=T0)
    assert len(COMMITS) == 2 and '"answered"' in COMMITS[1]


def test_refused_or_failed_asks_commit_nothing(mod, sent):
    def boom(_):
        raise RuntimeError("telegram down")

    _create(mod, "Q?", "why", now=T0, send=boom)
    _create(mod, "", "why", now=T0, send=sent.append)
    assert COMMITS == []


def test_quiet_turns_do_not_commit(mod, sent):
    assert mod._asks_context(DEPS[0], now=T0) == ""
    _ask(mod, sent)
    n = len(COMMITS)
    mod._asks_context(DEPS[0], now=T0 + timedelta(hours=1))  # open ask, nothing changed
    assert len(COMMITS) == n
    mod._asks_context(DEPS[0], now=T0 + timedelta(days=4))  # lapses: one write
    mod._asks_context(DEPS[0], now=T0 + timedelta(days=4, hours=1))  # already reported
    assert len(COMMITS) == n + 1


def test_commit_failure_keeps_the_ask(mod, sent, monkeypatch):
    class _Down:
        def commit(self):
            raise RuntimeError("volume busy")

    DEPS[0].storage = _Down()
    res = _ask(mod, sent)
    assert res["ok"] and _file(mod)["asks"][0]["id"] == res["id"]
