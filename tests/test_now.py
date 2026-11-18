"""The "now" report: report_status's writer and the freshness check.

Runs the real `agent/main.py` helpers against a temp now.json and an
in-memory stand-in for the status Dict. No network, no Modal.

Why: Rinkesh couldn't tell what the agent was doing without reading logs,
state.md and the Volume. state.md was meant to answer that but is prose the
agent rewrites when it remembers (12 hours stale on 11-17). These tests pin
what fixes that: a report needs the doing/why fields, is committed at once,
is tied to the turn it was written in, and is flagged as out of date when a
later run finishes without replacing it.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import modal  # noqa: E402

STORE = {}
T0 = datetime(2026, 11, 17, 22, 0, tzinfo=timezone.utc)


class _FakeDict:
    def __init__(self, name):
        self.name = name

    def get(self, key, default=None):
        return STORE.get((self.name, key), default)

    def put(self, key, value):
        STORE[(self.name, key)] = value


@pytest.fixture()
def mod(monkeypatch, tmp_path):
    STORE.clear()
    monkeypatch.setattr(
        modal.Dict, "from_name", classmethod(lambda cls, name, create_if_missing=False: _FakeDict(name))
    )
    sys.modules.pop("main", None)
    import main

    monkeypatch.setattr(main, "NOW_PATH", str(tmp_path / "memory" / "now.json"))
    main.COMMITS = []

    class _Volume:
        def commit(self):
            main.COMMITS.append(Path(main.NOW_PATH).read_text(encoding="utf-8"))

    monkeypatch.setattr(main, "volume", _Volume())
    return main


def _running(mod, since):
    STORE[(mod.STATUS_DICT_NAME, "current")] = {"state": "running-hourly", "since": since}


def test_report_requires_doing_and_why(mod):
    assert not mod._now_report("", "because")["ok"]
    assert not mod._now_report("fixing", "  ")["ok"]
    assert not Path(mod.NOW_PATH).exists()


def test_report_rejects_a_path_as_project(mod):
    res = mod._now_report("x", "y", project="../state")
    assert not res["ok"]
    assert mod._now_report("x", "y", project="gateway-ui-craft.md")["project"] == "gateway-ui-craft"


def test_report_is_written_and_committed_at_once(mod):
    _running(mod, "2026-11-17T21:55:00+00:00")
    res = mod._now_report(
        "Fixing how line_measure reads line width",
        'He said "close it, one bug first"',
        project="gateway-ui-craft",
        done_when="gutter live and the full check green at four widths",
        done=["gutter live", "  widths   added "],
        left=["fix line_measure", "full check", ""],
        next_step="full check",
        now=T0,
    )
    assert res["ok"]
    on_disk = json.loads(Path(mod.NOW_PATH).read_text(encoding="utf-8"))
    assert on_disk["turn_started"] == "2026-11-17T21:55:00+00:00"
    assert on_disk["reported_at"] == "2026-11-17T22:00:00+00:00"
    assert on_disk["done"] == ["gutter live", "widths added"]
    assert on_disk["left"] == ["fix line_measure", "full check"]
    assert len(mod.COMMITS) == 1 and json.loads(mod.COMMITS[0])["doing"].startswith("Fixing")


def test_report_caps_long_fields(mod):
    res = mod._now_report("d" * 5000, "w", done=[str(i) for i in range(50)], now=T0)
    assert len(res["doing"]) == mod.NOW_TEXT_MAX
    assert len(res["done"]) == mod.NOW_LIST_MAX


def test_report_outside_a_running_turn_has_no_turn(mod):
    STORE[(mod.STATUS_DICT_NAME, "current")] = {"state": "idle", "since": "2026-11-17T21:00:00+00:00"}
    assert mod._now_report("x", "y", now=T0)["turn_started"] is None


def _turn(started, ended="2026-11-17T23:59:00+00:00"):
    return {"started": started, "ended": ended, "outcome": "done"}


def test_fresh_when_no_later_run_finished(mod):
    rec = {"reported_at": "2026-11-17T22:00:00+00:00", "turn_started": "2026-11-17T21:55:00+00:00"}
    turns = [_turn("2026-11-17T21:00:00+00:00"), _turn("2026-11-17T21:55:00+00:00"),
             _turn("2026-11-17T23:00:00+00:00", ended=None)]  # running: not yet its chance
    assert mod._now_freshness(rec, turns)["fresh"]


def test_stale_when_later_runs_finished_without_reporting(mod):
    rec = {"reported_at": "2026-11-17T22:00:00+00:00", "turn_started": "2026-11-17T21:55:00+00:00"}
    turns = [_turn("2026-11-17T21:55:00+00:00"), _turn("2026-11-17T23:00:00+00:00"),
             _turn("2026-11-18T00:00:00+00:00")]
    f = mod._now_freshness(rec, turns)
    assert not f["fresh"] and f["missed_runs"] == 2


def test_report_without_turn_falls_back_to_its_time(mod):
    rec = {"reported_at": "2026-11-17T22:00:00+00:00", "turn_started": None}
    assert mod._now_freshness(rec, [_turn("2026-11-17T21:55:00+00:00")])["fresh"]
    assert not mod._now_freshness(rec, [_turn("2026-11-17T22:30:00+00:00")])["fresh"]


def test_no_report_is_not_fresh(mod):
    assert mod._now_freshness(None, [])["reason"] == "no report yet"
