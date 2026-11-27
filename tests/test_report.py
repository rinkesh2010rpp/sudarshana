"""report_status: the report stored on the turn record, and how /control reads it.

Runs the real `sudarshana.memory.report` helpers against an in-memory stand-in for the
status Dict. No network, no Modal.

Why: Rinkesh couldn't tell what the agent was doing without reading logs,
state.md and the Volume. state.md was meant to answer that but is prose the
agent rewrites when it remembers (12 hours stale on 11-17). These tests pin
what fixes that: a report needs the doing/why fields, lands on the running
turn's own record (so it always knows which run it came from), survives the
turn-end write, and is flagged as out of date when later runs finish
without one.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from conftest import fake_deps  # noqa: E402

from sudarshana.core import config  # noqa: E402
from sudarshana.memory import report, roadmap, status  # noqa: E402

STORE = {}
DEPS = [None]  # this test's fake Deps, set by the fixture


class _FakeDict:
    def __init__(self, name):
        self.name = name

    def get(self, key, default=None):
        return STORE.get(key, default)

    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in STORE:
            return
        STORE[key] = value

    def __setitem__(self, key, value):
        STORE[key] = value

    def __delitem__(self, key):
        del STORE[key]

    def keys(self):
        return list(STORE.keys())

    def items(self):
        return list(STORE.items())


@pytest.fixture()
def mod(monkeypatch, patch_all, tmp_path):
    STORE.clear()
    root = tmp_path.as_posix()
    patch_all("VOLUME_PATH", root)
    patch_all("ROADMAP_PATH", f"{root}/memory/roadmap.json")
    patch_all("ACTIONS_DIR", f"{root}/actions")
    patch_all("ACTIONS_ARCHIVE_DIR", f"{root}/actions/archive")

    DEPS[0] = fake_deps(_FakeDict)
    return report


def _roadmap(mod):
    """gate-axe with t1 done and t2 open; post-meta with no tasks."""
    roadmap._initiative_create(DEPS[0], "gate-axe", "Replace what axe covers", "o", "full green run on axe", status="active")
    roadmap._task_add(DEPS[0], "gate-axe", "Prove axe works", "planted defects named")
    roadmap._task_add(DEPS[0], "gate-axe", "Build axe_stage.py", "fails on planted defect")
    run = f"{config.VOLUME_PATH}/run.txt"
    open(run, "w").close()
    roadmap._task_update(DEPS[0], "gate-axe", "t1", status="doing")
    roadmap._task_update(DEPS[0], "gate-axe", "t1", status="done", deliverables=[run], handoff="t2 next")
    roadmap._initiative_create(DEPS[0], "post-meta", "Posts share as themselves", "o", "each post previews itself")


def _report(mod, key=None, **kw):
    if key is None:
        key = (STORE.get("current") or {}).get("_turn_key")
    args = {"doing": "Fixing line_measure", "why": 'He said "close it, one bug first"'}
    args.update(kw)
    return mod._report_save(DEPS[0], key, **args)


def test_report_requires_doing_and_why(mod):
    key = status._status_turn_start(DEPS[0], "running-hourly")
    assert not mod._report_save(DEPS[0], key, "", "because")["ok"]
    assert not mod._report_save(DEPS[0], key, "fixing", "  ")["ok"]
    assert "report" not in STORE[key]


def test_worked_on_is_checked_against_the_roadmap(mod):
    # A free-text project could silently name nothing; now an unknown id is
    # refused with the valid ones, so the agent corrects it in the same turn.
    status._status_turn_start(DEPS[0], "running-hourly")
    _roadmap(mod)
    res = _report(mod, worked_on=["gate_axe"])
    assert not res["ok"] and "no initiative 'gate_axe'" in res["error"] and "gate-axe" in res["error"]
    res = _report(mod, worked_on=["gate-axe/t9"])
    assert not res["ok"] and "t1, t2" in res["error"]
    assert not _report(mod, worked_on=["../state"])["ok"]
    res = _report(mod, worked_on=["gate-axe/t2", "post-meta", "gate-axe/t2", "gate-axe.md"])
    assert res["report"]["worked_on"] == ["gate-axe/t2", "post-meta", "gate-axe"]


def test_a_turn_that_wasnt_roadmap_work_reports_with_no_ids(mod):
    key = status._status_turn_start(DEPS[0], "running-hourly")
    assert _report(mod, doing="wrote the daily post")["ok"]
    assert STORE[key]["report"]["worked_on"] == []


def test_progress_is_filled_in_from_the_roadmap_not_typed(mod):
    _roadmap(mod)
    rep = mod._report_with_progress({"doing": "built it", "worked_on": ["gate-axe/t2", "post-meta"]},
                                    roadmap._roadmap_load())
    # The flat fields the /control page renders, from the first initiative.
    assert rep["project"] == "gate-axe" and rep["done_when"] == "full green run on axe"
    assert rep["done"] == ["Prove axe works"] and rep["left"] == ["Build axe_stage.py"]
    d = rep["worked_on_detail"]
    assert d[0]["task"] == {"id": "t2", "title": "Build axe_stage.py", "status": "todo"}
    assert d[0]["tasks_done"] == 1 and d[0]["tasks_open"] == 1
    assert d[1]["initiative"] == "post-meta" and "task" not in d[1]


def test_an_id_gone_from_the_roadmap_shows_as_missing(mod):
    rep = mod._report_with_progress({"doing": "x", "worked_on": ["gone/t1"]}, [])
    assert rep["worked_on_detail"] == [{"ref": "gone/t1", "initiative": "gone", "missing": True}]
    assert rep["project"] == "" and rep["done"] == [] and rep["left"] == []


def test_an_old_report_keeps_the_fields_it_typed(mod):
    _roadmap(mod)
    old = {"doing": "x", "project": "gateway-ui-craft", "done": ["a"], "left": ["b"], "done_when": "w"}
    rep = mod._report_with_progress(old, roadmap._roadmap_load())
    assert rep["project"] == "gateway-ui-craft" and rep["done"] == ["a"] and rep["done_when"] == "w"


def test_report_needs_a_running_turn(mod):
    res = _report(mod, key="")
    assert not res["ok"] and "no running turn" in res["error"]
    assert not _report(mod, key="turn:never-started")["ok"]


def test_overlapping_turns_each_keep_their_own_report(mod, patch_all):
    # A redeploy: the old container's turn is still running when the new one
    # starts, so `current` names the later turn. Each report must still land
    # on the turn that wrote it, and neither turn may show as missed.
    # Relative to now, not fixed dates: turn-end prunes records older than
    # STATUS_RECORDS_TTL_SECONDS, so fixed dates turned this test red once
    # they were 36 hours old (it pruned its own records mid-test).
    base = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=10)
    stamps = [(base + timedelta(seconds=s)).isoformat() for s in (0, 30, 300)]
    times = iter(stamps[:2])
    patch_all("_status_ts", lambda: next(times, stamps[2]))
    early = status._status_turn_start(DEPS[0], "running-hourly")
    late = status._status_turn_start(DEPS[0], "running-telegram")
    assert STORE["current"]["_turn_key"] == late
    assert _report(mod, key=early, doing="hourly work")["ok"]
    assert _report(mod, key=late, doing="answering his message")["ok"]
    assert STORE[early]["report"]["doing"] == "hourly work"
    assert STORE[late]["report"]["doing"] == "answering his message"
    status._status_turn_end(DEPS[0], early, "done")
    status._status_turn_end(DEPS[0], late, "done")
    turns = sorted((v for k, v in STORE.items() if k.startswith("turn:")), key=lambda r: r["started"])
    out = mod._report_latest(turns)
    assert out["fresh"] and out["missed_runs"] == 0 and out["doing"] == "answering his message"


def test_report_lands_on_the_running_turn_and_survives_turn_end(mod):
    key = status._status_turn_start(DEPS[0], "running-hourly")
    _roadmap(mod)
    res = _report(mod, worked_on=["gate-axe/t2"], next_step="  swap   structure-control ")
    assert res["ok"]
    status._status_turn_end(DEPS[0], key, "done", "hourly turn finished")
    rep = STORE[key]["report"]
    assert STORE[key]["ended"] is not None
    assert rep["worked_on"] == ["gate-axe/t2"] and rep["next"] == "swap structure-control"
    # Only what the agent wrote is stored; progress is read from the roadmap.
    assert not {"project", "done", "left", "done_when"} & set(rep)
    # Private: the public status API serves `history` and `current`, never the report.
    assert all("report" not in h for h in STORE["history"])
    assert "report" not in STORE["current"]


def test_report_caps_long_fields(mod):
    status._status_turn_start(DEPS[0], "running-hourly")
    rep = _report(mod, doing="d" * 5000, next_step="n" * 5000)["report"]
    assert len(rep["doing"]) == mod.REPORT_TEXT_MAX and len(rep["next"]) == mod.REPORT_TEXT_MAX


def _turn(started, report=None, ended="x"):
    t = {"started": started, "ended": ended, "outcome": "done", "state": "running-hourly"}
    if report:
        t["report"] = report
    return t


def test_latest_is_fresh_when_the_last_finished_run_reported(mod):
    turns = [_turn("1", {"doing": "a"}), _turn("2", {"doing": "b"}), _turn("3", ended=None)]
    out = mod._report_latest(turns)
    assert out["fresh"] and out["doing"] == "b" and out["run"]["kind"] == "hourly"


def test_latest_counts_finished_runs_that_did_not_report(mod):
    turns = [_turn("1", {"doing": "a"}), _turn("2"), _turn("3")]
    out = mod._report_latest(turns)
    assert not out["fresh"] and out["missed_runs"] == 2 and out["doing"] == "a"


def test_latest_with_no_report_at_all(mod):
    out = mod._report_latest([_turn("1")])
    assert out["run"] is None and "doing" not in out and not out["fresh"] and out["missed_runs"] == 1
