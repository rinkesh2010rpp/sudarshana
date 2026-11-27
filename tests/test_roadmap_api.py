"""/api/control/roadmap: the panel's view of the roadmap store.

Builds the real control_api FastAPI app with forward-auth stubbed to the
owner, against a temp Volume directory. No network, no Modal.

Why: /control could only serve ROADMAP.md raw, so the panel had to parse
prose to show status. These pin that it now gets records in roadmap order,
the same focus the agent's note picks, a doc link that is flagged when the
file is gone, and that a doc path can't be pointed outside the actions dir.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


import importlib  # noqa: E402

import modal  # noqa: E402

APP = "sudarshana.infra.runtime.modal.app"

from sudarshana.memory import roadmap  # noqa: E402

T0 = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
AUTH = {"Authorization": "Bearer t"}


class _FakeDict:
    def __init__(self, name):
        self.name = name

    def get(self, key, default=None):
        return default

    def put(self, key, value):
        pass

    def items(self):
        return []


class _Resp:
    status_code = 200

    def json(self):
        return {"email": "owner@example.com"}



def _deps(main):
    """The deps control_api builds, so writes land where it reads."""
    return main.make_deps()


@pytest.fixture()
def env(monkeypatch, patch_all, tmp_path):
    from fastapi.testclient import TestClient
    import requests

    monkeypatch.setattr(
        modal.Dict, "from_name", classmethod(lambda cls, name, create_if_missing=False: _FakeDict(name))
    )
    monkeypatch.setenv("CONTROL_OWNER_EMAIL", "owner@example.com")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    sys.modules.pop(APP, None)
    main = importlib.import_module(APP)

    root = tmp_path.as_posix()
    for name, val in {
        "VOLUME_PATH": root,
        "ROADMAP_PATH": f"{root}/memory/roadmap.json",
        "ASKS_PATH": f"{root}/memory/asks.json",
        "ACTIONS_DIR": f"{root}/actions",
        "ACTIONS_ARCHIVE_DIR": f"{root}/actions/archive",
    }.items():
        patch_all(name, val)

    class _Volume:
        def commit(self):
            pass

        def reload(self):
            pass

    patch_all("volume", _Volume())
    # .local() runs the undecorated function; get_raw_f() is gone in modal 1.6.
    app = main.control_api.local()
    return main, TestClient(app)


def test_roadmap_comes_as_records_in_order_with_the_focus(env):
    main, client = env
    roadmap._initiative_create(_deps(main), "later", "Later", "o", "d", status="active", priority=2, now=T0)
    roadmap._initiative_create(_deps(main), "first", "First", "o", "d", status="active", priority=1, now=T0)
    roadmap._task_add(_deps(main), "first", "A", "a done", now=T0)
    body = client.get("/api/control/roadmap", headers=AUTH).json()
    assert body["error"] is None
    assert [i["id"] for i in body["initiatives"]] == ["first", "later"]
    assert body["focus"] == "first"
    first = body["initiatives"][0]
    assert first["doc_exists"] is True and first["tasks_open"] == 1
    assert first["tasks"][0]["title"] == "A"


def test_one_initiative_and_its_doc(env):
    main, client = env
    roadmap._initiative_create(_deps(main), "gate-axe", "T", "objective here", "d", now=T0)
    assert client.get("/api/control/roadmap/gate-axe", headers=AUTH).json()["id"] == "gate-axe"
    doc = client.get("/api/control/roadmap/gate-axe/doc", headers=AUTH).json()
    assert "objective here" in doc["content"]
    assert client.get("/api/control/roadmap/nope", headers=AUTH).status_code == 404
    assert client.get("/api/control/roadmap/Bad%20Id", headers=AUTH).status_code == 400


def test_a_deleted_doc_shows_as_missing(env):
    main, client = env
    it = roadmap._initiative_create(_deps(main), "gate-axe", "T", "o", "d", now=T0)["initiative"]
    Path(it["doc"]).unlink()
    body = client.get("/api/control/roadmap", headers=AUTH).json()
    assert body["initiatives"][0]["doc_exists"] is False
    assert client.get("/api/control/roadmap/gate-axe/doc", headers=AUTH).status_code == 404


def test_a_doc_path_outside_actions_is_refused(env):
    main, client = env
    roadmap._initiative_create(_deps(main), "gate-axe", "T", "o", "d", now=T0)
    p = Path(roadmap.ROADMAP_PATH)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["initiatives"][0]["doc"] = roadmap.ROADMAP_PATH  # hand-edited to point elsewhere
    p.write_text(json.dumps(data), encoding="utf-8")
    assert client.get("/api/control/roadmap/gate-axe/doc", headers=AUTH).status_code == 400


def test_a_broken_store_is_an_error_not_an_empty_roadmap(env):
    main, client = env
    p = Path(roadmap.ROADMAP_PATH)
    p.parent.mkdir(parents=True)
    p.write_text("{bad", encoding="utf-8")
    body = client.get("/api/control/roadmap", headers=AUTH).json()
    assert body["initiatives"] == [] and body["error"]


def test_needs_the_owner(env):
    _, client = env
    assert client.get("/api/control/roadmap").status_code == 401


def test_overview_report_gets_its_progress_from_the_roadmap(env, monkeypatch):
    main, client = env
    roadmap._initiative_create(_deps(main), "gate-axe", "Replace what axe covers", "o", "full green run", status="active", now=T0)
    roadmap._task_add(_deps(main), "gate-axe", "Build axe_stage.py", "fails on planted defect", now=T0)
    stored = {"doing": "built it", "why": "he asked", "worked_on": ["gate-axe/t1"], "next": "swap",
              "run": {"kind": "hourly"}, "fresh": True, "missed_runs": 0}
    monkeypatch.setattr(main, "_report_latest", lambda recs: dict(stored))
    rep = client.get("/api/control/overview", headers=AUTH).json()["report"]
    assert rep["project"] == "gate-axe" and rep["done_when"] == "full green run"
    assert rep["left"] == ["Build axe_stage.py"] and rep["done"] == []
    assert rep["worked_on_detail"][0]["task"]["title"] == "Build axe_stage.py"
    assert rep["roadmap_error"] is None
