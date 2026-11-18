"""report_status: the report stored on the turn record, and how /control reads it.

Runs the real `agent/main.py` helpers against an in-memory stand-in for the
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
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import modal  # noqa: E402

STORE = {}


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
def mod(monkeypatch):
    STORE.clear()
    monkeypatch.setattr(
        modal.Dict, "from_name", classmethod(lambda cls, name, create_if_missing=False: _FakeDict(name))
    )
    sys.modules.pop("main", None)
    import main

    return main


def _report(mod, **kw):
    args = {"doing": "Fixing line_measure", "why": 'He said "close it, one bug first"'}
    args.update(kw)
    return mod._report_save(**args)


def test_report_requires_doing_and_why(mod):
    key = mod._status_turn_start("running-hourly")
    assert not mod._report_save("", "because")["ok"]
    assert not mod._report_save("fixing", "  ")["ok"]
    assert "report" not in STORE[key]


def test_report_rejects_a_path_as_project(mod):
    mod._status_turn_start("running-hourly")
    assert not _report(mod, project="../state")["ok"]
    assert _report(mod, project="gateway-ui-craft.md")["report"]["project"] == "gateway-ui-craft"


def test_report_needs_a_running_turn(mod):
    res = _report(mod)
    assert not res["ok"] and "no running turn" in res["error"]


def test_report_lands_on_the_running_turn_and_survives_turn_end(mod):
    key = mod._status_turn_start("running-hourly")
    res = _report(
        mod,
        project="gateway-ui-craft",
        done_when="gutter live and the full check green at four widths",
        done=["gutter live", "  widths   added "],
        left=["fix line_measure", "full check", ""],
        next_step="full check",
    )
    assert res["ok"]
    mod._status_turn_end(key, "done", "hourly turn finished")
    rep = STORE[key]["report"]
    assert STORE[key]["ended"] is not None
    assert rep["done"] == ["gutter live", "widths added"]
    assert rep["left"] == ["fix line_measure", "full check"]
    # Private: the public status API serves `history` and `current`, never the report.
    assert all("report" not in h for h in STORE["history"])
    assert "report" not in STORE["current"]


def test_report_caps_long_fields(mod):
    mod._status_turn_start("running-hourly")
    rep = _report(mod, doing="d" * 5000, done=[str(i) for i in range(50)])["report"]
    assert len(rep["doing"]) == mod.REPORT_TEXT_MAX
    assert len(rep["done"]) == mod.REPORT_LIST_MAX


def _turn(started, report=None, ended="x"):
    t = {"started": started, "ended": ended, "outcome": "done", "state": "running-hourly"}
    if report:
        t["report"] = report
    return t


def test_latest_is_fresh_when_the_last_finished_run_reported(mod):
    turns = [_turn("1", {"doing": "a"}), _turn("2", {"doing": "b"}), _turn("3", ended=None)]
    out = mod._report_latest(turns)
    assert out["fresh"] and out["report"]["doing"] == "b" and out["run"]["kind"] == "hourly"


def test_latest_counts_finished_runs_that_did_not_report(mod):
    turns = [_turn("1", {"doing": "a"}), _turn("2"), _turn("3")]
    out = mod._report_latest(turns)
    assert not out["fresh"] and out["missed_runs"] == 2 and out["report"]["doing"] == "a"


def test_latest_with_no_report_at_all(mod):
    out = mod._report_latest([_turn("1")])
    assert out["report"] is None and not out["fresh"] and out["missed_runs"] == 1
