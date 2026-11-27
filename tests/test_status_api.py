"""/api/status and /api/events: the public, read-only turn status.

Builds the real status app (interfaces/http/status.py) on fake deps. No
network, no Modal.

Why: this endpoint is public, so it must never show the internal keys the
writer keeps on `current` (like _turn_key), and /api/events must stay
bounded however large a limit is asked for.
"""
from fastapi.testclient import TestClient

from conftest import fake_deps
from sudarshana.infra.storage.local.stores import DictKV
from sudarshana.interfaces.http.status import create_status_app
from sudarshana.memory.status import STATUS_HISTORY_LIMIT


def _client():
    deps = fake_deps(lambda name: DictKV())
    kv = deps.status_kv
    kv["current"] = {"state": "running-hourly", "since": "t0", "turn_id": "x", "_turn_key": "turn:t0"}
    kv["history"] = [{"turn": i} for i in range(STATUS_HISTORY_LIMIT + 10)]
    kv["current_trace"] = [{"event": "turn_started"}]
    return TestClient(create_status_app(deps))


def test_status_hides_internal_keys():
    body = _client().get("/api/status").json()
    assert body["current"] == {"state": "running-hourly", "since": "t0", "turn_id": "x"}
    assert body["last_turn"] == {"turn": 0}


def test_events_limit_is_capped():
    client = _client()
    assert len(client.get("/api/events?limit=10000").json()["events"]) == STATUS_HISTORY_LIMIT
    assert len(client.get("/api/events?limit=0").json()["events"]) == 1
