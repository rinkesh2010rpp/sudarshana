"""The Telegram webhook: sender gate, update_id dedup, and spawning the turn.

Runs the real `agent/main.py` against an in-memory stand-in for `modal.Dict`
and a recording stand-in for the spawned `process_message`, so what is tested
is the code that ships. No network, no Modal, no model.

Why these cases: on 2026-11-11 one message became four identical turns. The
webhook sat in the same single container as a long turn, Telegram timed out
waiting for its ack and resent the update, and each resend started a turn.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import modal  # noqa: E402

STORE = {}


class _FakeDict:
    def __init__(self, name):
        self.name = name

    def get(self, key, default=None):
        return STORE.get((self.name, key), default)

    def put(self, key, value):
        STORE[(self.name, key)] = value


@pytest.fixture()
def mod(monkeypatch):
    """The module, with modal.Dict and the spawned method faked out."""
    STORE.clear()
    monkeypatch.setattr(
        modal.Dict, "from_name", classmethod(lambda cls, name, create_if_missing=False: _FakeDict(name))
    )
    for name in [m for m in sys.modules if m in ("main",)]:
        del sys.modules[name]
    import main

    spawned = []

    class _Spawn:
        def spawn(self, text, update_id=None):
            spawned.append((text, update_id))

    class _Sudarshana:
        def __init__(self):
            self.process_message = _Spawn()

    monkeypatch.setattr(main, "Sudarshana", _Sudarshana)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_ID", "4242")
    main.spawned = spawned
    main.webhook = main.telegram_webhook.get_raw_f()
    return main


def _update(update_id, text="hello", sender=4242):
    return {"update_id": update_id, "message": {"text": text, "from": {"id": sender}}}


def test_a_message_spawns_one_turn(mod):
    assert mod.webhook(_update(1)) == {"ok": True}
    assert mod.spawned == [("hello", 1)]


def test_a_resent_update_spawns_no_second_turn(mod):
    """The bug itself: Telegram resends; only one turn may start."""
    for _ in range(4):
        assert mod.webhook(_update(7))["ok"] is True
    assert mod.spawned == [("hello", 7)]
    assert mod.webhook(_update(7)).get("duplicate") is True


def test_a_later_message_still_gets_through(mod):
    mod.webhook(_update(10, "first"))
    mod.webhook(_update(11, "second"))
    assert [t for t, _ in mod.spawned] == ["first", "second"]


def test_other_senders_are_dropped_silently(mod):
    assert mod.webhook(_update(20, sender=1)) == {"ok": True}
    assert mod.spawned == []


def test_non_text_updates_are_ignored(mod):
    assert mod.webhook({"update_id": 21, "edited_message": {"text": "x"}}) == {"ok": True}
    assert mod.spawned == []


def test_malformed_payload_does_not_raise(mod):
    """A 500 here makes Telegram retry forever — the ack must always come."""
    assert mod.webhook({}) == {"ok": True}
    assert mod.webhook({"update_id": 22, "message": {"text": "hi"}}) == {"ok": True}
    assert mod.spawned == []


def test_missing_update_id_still_spawns(mod):
    """No id, no dedup — but never drop a real message over it."""
    mod.webhook({"message": {"text": "hi", "from": {"id": 4242}}})
    assert mod.spawned == [("hi", None)]


def test_claim_window_is_bounded(mod):
    for uid in range(1000, 1000 + mod.TELEGRAM_CLAIM_WINDOW + 50):
        mod.webhook(_update(uid, "bulk"))
    seen = STORE[("sudarshana-telegram-claims", "seen:webhook")]
    assert len(seen) == mod.TELEGRAM_CLAIM_WINDOW


def test_the_two_gates_do_not_consume_each_others_claim(mod):
    """The webhook claims "webhook"; the turn claims "run" for the same id."""
    assert mod._telegram_claim(5555, "webhook") is False
    assert mod._telegram_claim(5555, "run") is False
    assert mod._telegram_claim(5555, "run") is True
