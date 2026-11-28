"""The single-machine server runtime (infra/runtime/server/app.py).

Drives the real HTTP app and worker with fake deps and the turns stubbed.
No model, no network, no Modal.

Why: this runtime replaces Modal's spawn and single container with a queue
and one worker thread. These pin that a genuine Telegram update is queued
(not run on the request), a forged one is not, the status API is served
alongside, and the worker runs queued turns and survives one that fails.
"""
import queue

from fastapi.testclient import TestClient

from conftest import fake_deps
from sudarshana.infra.runtime.server import app as server
from sudarshana.infra.storage.local.stores import DictKV

SECRET = "s3cret"


def _update(uid, text="hi", sender=4242):
    return {"update_id": uid, "message": {"text": text, "from": {"id": sender}}}


def _client(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_ID", "4242")
    queued = []
    deps = fake_deps(lambda name: DictKV())
    return TestClient(server.create_app(deps, queued.append)), queued


def test_a_genuine_update_is_queued_not_run(monkeypatch):
    client, queued = _client(monkeypatch)
    r = client.post("/telegram", json=_update(1), headers={"X-Telegram-Bot-Api-Secret-Token": SECRET})
    assert r.json() == {"ok": True}
    assert queued == [("telegram", "hi", 1)]


def test_a_forged_update_is_dropped(monkeypatch):
    client, queued = _client(monkeypatch)
    r = client.post("/telegram", json=_update(2), headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"})
    assert r.status_code == 200 and queued == []


def test_the_status_api_is_served_too(monkeypatch):
    client, _ = _client(monkeypatch)
    assert client.get("/api/status").status_code == 200


def test_the_worker_runs_turns_in_order_and_survives_a_failure(monkeypatch):
    ran = []

    def fake_telegram_turn(agent, deps, text, update_id):
        if text == "boom":
            raise RuntimeError("turn failed")
        ran.append(text)

    monkeypatch.setattr(server, "telegram_turn", fake_telegram_turn)
    monkeypatch.setattr(server, "hourly_turn", lambda agent, deps: ran.append("hourly"))
    jobs = queue.Queue()
    for job in [("telegram", "a", 1), ("telegram", "boom", 2), ("hourly",), ("telegram", "b", 3)]:
        jobs.put(job)
    import threading

    threading.Thread(target=server.run_worker, args=(object(), None, jobs), daemon=True).start()
    jobs.join()
    assert ran == ["a", "hourly", "b"]
