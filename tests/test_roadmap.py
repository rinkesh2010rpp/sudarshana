"""The roadmap store: initiatives -> tasks, written only by the roadmap tools.

Runs the real `sudarshana.memory.roadmap` helpers against a temp Volume directory. No
network, no Modal.

Why: ROADMAP.md had grown to ~1,100 lines where an initiative's status was a
word in a heading, every cycle started by reading it, and /control could only
serve it raw. These tests pin what the store adds: status moves are checked
(nothing closed reopens, nothing waits on nothing), every initiative links to
a doc that exists and follows it into the archive, and one task is in
progress at a time with a handoff the next cycle reads. Statuses are the
agent's own to manage: closing an ask changes nothing on the roadmap.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import modal  # noqa: E402

from sudarshana.memory import asks, roadmap  # noqa: E402

COMMITS = []

T0 = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)


class _FakeDict:
    def __init__(self, name):
        self.name = name

    def get(self, key, default=None):
        return default

    def put(self, key, value):
        pass


@pytest.fixture()
def mod(monkeypatch, patch_all, tmp_path):
    monkeypatch.setattr(
        modal.Dict, "from_name", classmethod(lambda cls, name, create_if_missing=False: _FakeDict(name))
    )
    root = tmp_path.as_posix()
    patch_all("VOLUME_PATH", root)
    patch_all("ROADMAP_PATH", f"{root}/memory/roadmap.json")
    patch_all("ASKS_PATH", f"{root}/memory/asks.json")
    patch_all("ACTIONS_DIR", f"{root}/actions")
    patch_all("ACTIONS_ARCHIVE_DIR", f"{root}/actions/archive")
    COMMITS.clear()

    class _Volume:
        def commit(self):
            COMMITS.append(1)

    patch_all("volume", _Volume())
    return roadmap


def _store(mod):
    return json.loads(Path(mod.ROADMAP_PATH).read_text(encoding="utf-8"))["initiatives"]


def _new(mod, iid="gate-axe", **kw):
    args = dict(title="Replace what axe covers", objective="Swap stages axe can do",
                done_when="Full green run on the axe stack", now=T0)
    args.update(kw)
    return mod._initiative_create(iid, **args)


def _active(mod, iid="gate-axe", **kw):
    res = _new(mod, iid, status="active", **kw)
    assert res["ok"], res
    return res["initiative"]


def _deliverable(mod, name="run.txt"):
    p = Path(mod.VOLUME_PATH) / "gate-runs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok", encoding="utf-8")
    return p.as_posix()


def _ask(mod, q="Adopt gate-axe?"):
    res = asks._ask_create(q, "needs his yes", now=T0, send=lambda text: None)
    assert res["ok"], res
    return res["id"]


# --- initiatives ---------------------------------------------------------------

def test_create_starts_proposed_and_writes_its_doc(mod):
    res = _new(mod)
    assert res["ok"]
    it = res["initiative"]
    assert it["status"] == "proposed"
    assert it["doc"].endswith("actions/gate-axe.md")
    doc = Path(it["doc"]).read_text(encoding="utf-8")
    assert "Swap stages axe can do" in doc and "## Notes" in doc
    assert _store(mod)[0]["id"] == "gate-axe"
    assert len(COMMITS) == 1  # committed at once, not left to turn end


def test_existing_doc_is_kept_as_it_is(mod):
    p = Path(mod.ACTIONS_DIR) / "gate-axe.md"
    p.parent.mkdir(parents=True)
    p.write_text("# my old notes\n", encoding="utf-8")
    assert _new(mod)["ok"]
    assert p.read_text(encoding="utf-8") == "# my old notes\n"


def test_create_rejects_bad_ids_missing_fields_and_duplicates(mod):
    assert not _new(mod, "Gate Axe")["ok"]
    assert not _new(mod, done_when="")["ok"]
    assert _new(mod)["ok"]
    assert "already exists" in _new(mod)["error"]


def test_status_moves_are_checked(mod):
    _new(mod)
    res = mod._initiative_update("gate-axe", status="done", outcome="x", now=T0)
    assert not res["ok"] and "proposed -> done" in res["error"]
    assert "status must be" in mod._initiative_update("gate-axe", status="bogus", now=T0)["error"]


def test_blocked_needs_a_reason_and_closing_needs_an_outcome(mod):
    _active(mod)
    assert "reason" in mod._initiative_update("gate-axe", status="blocked", now=T0)["error"]
    assert "outcome" in mod._initiative_update("gate-axe", status="done", now=T0)["error"]
    res = mod._initiative_update("gate-axe", status="blocked", reason="Netlify down", now=T0)
    assert res["initiative"]["blocked_reason"] == "Netlify down"


def test_closing_moves_the_doc_to_the_archive_and_keeps_the_link(mod):
    it = _active(mod)
    old = Path(it["doc"])
    res = mod._initiative_update("gate-axe", status="done", outcome="11/11 green on axe", now=T0)
    new = Path(res["initiative"]["doc"])
    assert not old.exists() and new.exists()
    assert new.parent.name == "archive"
    assert res["initiative"]["closed"] == T0.isoformat()
    # Final: a closed initiative doesn't change any more.
    assert "closed" in mod._initiative_update("gate-axe", title="x", now=T0)["error"]


def test_archiving_never_overwrites_an_older_archived_doc(mod):
    arch = Path(mod.ACTIONS_ARCHIVE_DIR)
    arch.mkdir(parents=True)
    (arch / "gate-axe.md").write_text("older", encoding="utf-8")
    _active(mod)
    res = mod._initiative_update("gate-axe", status="declined", outcome="dropped", now=T0)
    assert Path(res["initiative"]["doc"]).name == "gate-axe-2026-11-18.md"
    assert (arch / "gate-axe.md").read_text(encoding="utf-8") == "older"


def test_links_must_exist_or_be_urls(mod):
    res = _new(mod, links=[f"{mod.VOLUME_PATH}/nope.txt"])
    assert not res["ok"] and "does not exist" in res["error"]
    res = _new(mod, links=["/etc/passwd"])
    assert not res["ok"]
    res = _new(mod, links=[_deliverable(mod), "https://github.com/x/y/pull/1"])
    assert res["ok"] and len(res["initiative"]["links"]) == 2


# --- tasks ---------------------------------------------------------------------

def test_tasks_queue_in_order_and_take_a_position(mod):
    _active(mod)
    mod._task_add("gate-axe", "Build axe_stage.py", "fails on planted defect", now=T0)
    mod._task_add("gate-axe", "Swap structure-control", "green run", now=T0)
    res = mod._task_add("gate-axe", "Settle the map", "map agreed", position=1, now=T0)
    assert res["task"]["id"] == "t3"
    assert [t["id"] for t in _store(mod)[0]["tasks"]] == ["t3", "t1", "t2"]


def test_one_task_doing_at_a_time_and_only_on_an_active_initiative(mod):
    _new(mod)
    mod._task_add("gate-axe", "A", "a done", now=T0)
    mod._task_add("gate-axe", "B", "b done", now=T0)
    assert "active" in mod._task_update("gate-axe", "t1", status="doing", now=T0)["error"]
    mod._initiative_update("gate-axe", status="active", now=T0)
    assert mod._task_update("gate-axe", "t1", status="doing", now=T0)["ok"]
    res = mod._task_update("gate-axe", "t2", status="doing", now=T0)
    assert not res["ok"] and "t1 is already doing" in res["error"]


def test_done_needs_what_it_produced_and_a_handoff(mod):
    _active(mod)
    mod._task_add("gate-axe", "A", "a done", now=T0)
    mod._task_update("gate-axe", "t1", status="doing", now=T0)
    assert "deliverables" in mod._task_update("gate-axe", "t1", status="done", handoff="x", now=T0)["error"]
    res = mod._task_update("gate-axe", "t1", status="done", deliverables=[f"{mod.VOLUME_PATH}/missing"], handoff="x", now=T0)
    assert "does not exist" in res["error"]
    run = _deliverable(mod)
    assert "handoff" in mod._task_update("gate-axe", "t1", status="done", deliverables=[run], now=T0)["error"]
    res = mod._task_update("gate-axe", "t1", status="done", deliverables=[run], handoff="t2 next", now=T0)
    assert res["ok"] and res["task"]["deliverables"] == [run]


def test_handoff_is_kept_between_cycles(mod):
    _active(mod)
    mod._task_add("gate-axe", "A", "a done", now=T0)
    mod._task_update("gate-axe", "t1", status="doing", handoff="rerun batch B with --only states", now=T0)
    ctx = mod._roadmap_context(now=T0)
    assert "Handoff from last cycle: rerun batch B with --only states" in ctx


# --- waiting, and the agent's own control of status -------------------------------

def test_the_agent_adopts_and_starts_initiatives_itself(mod):
    # No approval field: the go-ahead rule lives in the prompt, not in code.
    assert _new(mod)["ok"]
    res = mod._initiative_update("gate-axe", status="active", now=T0)
    assert res["ok"] and "approval" not in res["initiative"]
    assert _new(mod, "post-meta", status="active")["ok"]


def test_waiting_says_what_on_in_free_text(mod):
    _active(mod)
    assert "blocked_on" in mod._initiative_update("gate-axe", status="waiting", now=T0)["error"]
    res = mod._initiative_update("gate-axe", status="waiting", blocked_on="ask 3f2a", now=T0)
    assert res["ok"] and res["initiative"]["blocked_on"] == "ask 3f2a"
    mod._task_add("gate-axe", "A", "a done", now=T0)


def test_closing_an_ask_changes_nothing_on_the_roadmap(mod):
    _active(mod)
    mod._task_add("gate-axe", "A", "a done", now=T0)
    mod._task_update("gate-axe", "t1", status="doing", handoff="keep this", now=T0)
    ask_id = _ask(mod)
    mod._task_update("gate-axe", "t1", status="waiting", blocked_on=f"ask {ask_id}", now=T0)
    before = _store(mod)
    res = asks._ask_resolve(ask_id, "use 4.12", now=T0)
    assert res == {"ok": True, "id": ask_id, "status": "answered"}
    assert _store(mod) == before
    # The agent moves it on itself.
    assert mod._task_update("gate-axe", "t1", status="doing", handoff="he said 4.12", now=T0)["ok"]


def test_a_waiting_initiative_is_skipped_until_the_agent_resumes_it(mod):
    _active(mod, "first", priority=0)
    _active(mod, "second", priority=1)
    mod._initiative_update("first", status="waiting", blocked_on="ask 3f2a", now=T0)
    ctx = mod._roadmap_context(now=T0)
    assert "FOCUS — second" in ctx and "first [waiting]" in ctx and "waiting on ask 3f2a" in ctx
    assert "move it on yourself" in ctx
    mod._initiative_update("first", status="active", now=T0)
    assert "FOCUS — first" in mod._roadmap_context(now=T0)


def test_no_cap_and_no_cycle_counting(mod):
    it = _active(mod)
    assert not {"max_cycles", "cycles_used"} & set(it)
    assert not hasattr(mod, "_roadmap_count_cycle")

# --- the per-turn note and a broken store ----------------------------------------

def test_empty_store_asks_for_the_migration(mod):
    assert "store is empty" in mod._roadmap_context(now=T0)


def test_focus_shows_objective_task_queue_and_cycle(mod):
    _active(mod)
    _new(mod, "post-meta")
    mod._task_add("gate-axe", "Build axe_stage.py", "fails on planted defect", now=T0)
    mod._task_add("gate-axe", "Swap structure-control", "green run", now=T0)
    ctx = mod._roadmap_context(now=T0)
    focus_line = next(line for line in ctx.splitlines() if line.startswith("FOCUS"))
    assert "gate-axe" in focus_line and "cycle" not in focus_line
    assert "Current task: t1 [todo] Build axe_stage.py" in ctx
    assert "t2 [todo] Swap structure-control" in ctx
    assert "post-meta [proposed]" in ctx and "not yet approved" in ctx


def test_a_broken_store_changes_nothing_and_the_note_fails_open(mod):
    p = Path(mod.ROADMAP_PATH)
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    res = _new(mod)
    assert not res["ok"] and "nothing changed" in res["error"]
    assert p.read_text(encoding="utf-8") == "{not json"
    assert "could not be read" in mod._roadmap_context(now=T0)


def test_view_lists_everything_and_one_in_full(mod):
    _active(mod)
    mod._task_add("gate-axe", "A", "a done", now=T0)
    assert "gate-axe [active]" in mod._roadmap_view()
    full = mod._roadmap_view("gate-axe")
    assert "Doc:" in full and "t1 [todo] A" in full
    assert "no initiative" in mod._roadmap_view("nope")
