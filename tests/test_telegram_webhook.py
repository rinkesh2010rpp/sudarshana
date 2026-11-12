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

    spawner = _Spawn()

    class _Sudarshana:
        # One shared spawner: the webhook builds a fresh handle per call, so a
        # test has to be able to reach the object those handles delegate to.
        def __init__(self):
            self.process_message = spawner

    monkeypatch.setattr(main, "Sudarshana", _Sudarshana)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_ID", "4242")
    main.spawned = spawned
    main.spawner = spawner
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
    """The webhook records "webhook"; the turn records "run" for the same id."""
    mod._telegram_record(5555, "webhook")
    assert mod._telegram_seen(5555, "webhook") is True
    assert mod._telegram_seen(5555, "run") is False
    mod._telegram_record(5555, "run")
    assert mod._telegram_seen(5555, "run") is True


def test_a_failed_spawn_loses_nothing(mod):
    """The bug Rinkesh caught: record first and a failed spawn eats the message.

    Telegram retries only if we answer 5xx; if we had already recorded the
    update_id, every retry would be dismissed as a duplicate and the message
    would be gone for good.
    """
    mod.webhook(_update(31, "hi"))

    def _boom(text, update_id=None):
        raise RuntimeError("no capacity")

    mod.spawner.spawn = _boom
    with pytest.raises(RuntimeError):
        mod.webhook(_update(32, "important"))

    # Nothing was recorded, so Telegram's retry is not treated as a duplicate.
    assert mod._telegram_seen(32, "webhook") is False
    assert STORE[("sudarshana-telegram-claims", "seen:webhook")] == [31]

    del mod.spawner.spawn
    assert mod.webhook(_update(32, "important")) == {"ok": True}
    assert mod.spawned[-1] == ("important", 32)


def test_a_dict_outage_does_not_break_the_webhook(mod, monkeypatch):
    """A Dict error must not become a 500, and must not drop the message."""
    def _down(cls, name, create_if_missing=False):
        raise RuntimeError("dict unavailable")

    monkeypatch.setattr(modal.Dict, "from_name", classmethod(_down))
    assert mod.webhook(_update(40, "still arrives")) == {"ok": True}
    assert mod.spawned == [("still arrives", 40)]


def test_a_dict_outage_cannot_kill_a_turn_that_was_already_acked(mod, monkeypatch):
    """Same outage, from inside process_message: the webhook already said ok,
    so raising here would discard the message with nothing left to retry."""
    def _down(cls, name, create_if_missing=False):
        raise RuntimeError("dict unavailable")

    monkeypatch.setattr(modal.Dict, "from_name", classmethod(_down))
    assert mod._telegram_seen(41, "run") is False
    mod._telegram_record(41, "run")  # must not raise


def test_unset_allowed_user_is_a_5xx_not_a_silent_drop(mod, monkeypatch):
    """A missing secret is misconfiguration: keep the update, don't discard it."""
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_ID", raising=False)
    response = mod.webhook(_update(50, "hello"))
    assert response.status_code == 503
    assert mod.spawned == []

