"""The per-turn report and its roadmap progress."""
from sudarshana.memory.roadmap import INITIATIVE_OPEN, TASK_OPEN, _rm_find, _rm_sorted, _rm_task, _roadmap_load
from sudarshana.memory.status import _status_dict, _status_ts

# --- report: what I'm doing, why, and where things stand ----------------------
# Rinkesh reads this on /control to see what the agent is doing, why, and how
# far along the work is ("does it need me" is asks.json). state.md was meant to
# answer that but is prose the agent rewrites when it remembers; it sat 12
# hours stale on 11-17. So the agent reports through a tool at the end of every
# turn, and the report is stored ON that turn's record in the status Dict —
# the same record /control's overview already reads. A report therefore
# always knows which run it came from, and a finished run with no report is
# visible as exactly that. Turn records are private: the public status API
# serves only `current` and `history`, never turn:* records.
REPORT_TEXT_MAX = 400
REPORT_LIST_MAX = 8


def _report_save(turn_key: str, doing: str, why: str, worked_on=None, next_step: str = "") -> dict:
    """Validate the report and write it onto `turn_key`'s record — the turn
    making the call, handed in by the caller. Not current._turn_key: that is
    whichever turn started last, so when two turns overlap (a redeploy, where
    the old container finishes a run while the new one starts) the earlier
    turn's report would land on the later turn's record.

    The agent writes only what code can't know: what it did, why, what's
    next, and which initiatives/tasks it worked on (checked against the
    roadmap store). Progress — titles, statuses, done_when, task counts — is
    filled in from the store when /control reads the report, so it can't go
    stale. Returns {"ok": True, "report": ...} or {"ok": False, "error": ...}."""
    doing = " ".join(str(doing or "").split())
    why = " ".join(str(why or "").split())
    if not doing:
        return {"ok": False, "error": "doing= is required: one sentence on what this turn did or is doing"}
    if not why:
        return {"ok": False, "error": "why= is required: the reason, in terms of what Rinkesh asked for"}
    if isinstance(worked_on, str):
        worked_on = [worked_on]
    refs = []
    if worked_on:
        try:
            items = _roadmap_load()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"can't read the roadmap store to check worked_on ({e})"}
        refs, err = _roadmap_refs(worked_on, items)
        if err:
            return {"ok": False, "error": err}

    report = {
        "reported_at": _status_ts(),
        "doing": doing[:REPORT_TEXT_MAX],
        "why": why[:REPORT_TEXT_MAX],
        "worked_on": refs,
        "next": " ".join(str(next_step or "").split())[:REPORT_TEXT_MAX],
    }
    try:
        d = _status_dict()
        rec = d.get(turn_key) if turn_key else None
        if not rec:
            return {"ok": False, "error": "no running turn to attach the report to"}
        rec["report"] = report
        d.put(turn_key, rec)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"status store write failed: {type(e).__name__}"}
    return {"ok": True, "report": report}


def _report_latest(turns: list) -> dict:
    """From turn records sorted by `started`: the latest report, flattened,
    plus `run` (the run that wrote it), `fresh` and `missed_runs`. A run
    still in progress hasn't had its chance to report and is skipped; every
    finished run after the report that has none is counted as missed. With
    no report in the records: {"run": None, "fresh": False, "missed_runs": n}."""
    missed = 0
    for t in reversed(turns or []):
        if t.get("ended") is None:
            continue
        rep = t.get("report")
        if rep:
            return {
                **rep,
                "run": {
                    "kind": str(t.get("state") or "").replace("running-", ""),
                    "started": t.get("started"),
                    "ended": t.get("ended"),
                    "outcome": t.get("outcome"),
                },
                "fresh": missed == 0,
                "missed_runs": missed,
            }
        missed += 1
    return {"run": None, "fresh": False, "missed_runs": missed}



# --- report progress, read from the roadmap ---------------------------------

def _roadmap_refs(refs, items: list):
    """Check report_status's worked_on: each entry is 'initiative' or
    'initiative/task' and must exist in the store. Returns (clean list, None)
    or (None, error naming the valid ids, so the agent can correct it)."""
    out = []
    for raw in refs or []:
        ref = str(raw or "").strip().removesuffix(".md")
        if not ref:
            continue
        iid, _, tid = ref.partition("/")
        it = _rm_find(items, iid)
        if not it:
            known = ", ".join(i["id"] for i in _rm_sorted(items) if i["status"] in INITIATIVE_OPEN) or "none"
            return None, f"worked_on: no initiative {iid!r}. Open initiatives: {known}"
        if tid and not _rm_task(it, tid):
            known = ", ".join(t["id"] for t in it.get("tasks") or []) or "none"
            return None, f"worked_on: {iid} has no task {tid!r}. Its tasks: {known}"
        ref = f"{it['id']}/{tid}" if tid else it["id"]
        if ref not in out:
            out.append(ref)
    return out[:REPORT_LIST_MAX], None


def _roadmap_report_detail(worked_on, items: list) -> list:
    """What /control shows for a report's worked_on: each entry with its
    initiative's live title, status and done_when, task counts, and the task's
    own title and status. Filled in when read, so it is never stale; an id
    that no longer exists is shown as missing rather than dropped."""
    out = []
    for ref in worked_on or []:
        iid, _, tid = str(ref).partition("/")
        it = _rm_find(items, iid)
        if not it:
            out.append({"ref": ref, "initiative": iid, "missing": True})
            continue
        tasks = it.get("tasks") or []
        row = {
            "ref": ref,
            "initiative": it["id"],
            "title": it["title"],
            "status": it["status"],
            "done_when": it["done_when"],
            "tasks_done": sum(1 for t in tasks if t["status"] == "done"),
            "tasks_open": sum(1 for t in tasks if t["status"] in TASK_OPEN),
            "missing": False,
        }
        if tid:
            t = _rm_task(it, tid)
            row["task"] = {"id": tid, "title": t["title"], "status": t["status"]} if t else {"id": tid, "missing": True}
        out.append(row)
    return out


def _report_with_progress(report: dict, items: list) -> dict:
    """A stored report plus its progress, read from the roadmap now:
    `worked_on_detail` for each worked_on entry, and the flat fields the
    /control page already renders — project, done_when, done, left — taken
    from the first initiative in worked_on (done/left are its task titles).
    Derived here, never typed by the agent, so they can't go stale."""
    out = dict(report or {})
    refs = out.get("worked_on") or []
    detail = _roadmap_report_detail(refs, items)
    out["worked_on_detail"] = detail
    first = next((d for d in detail if not d.get("missing")), None)
    it = _rm_find(items, first["initiative"]) if first else None
    tasks = (it or {}).get("tasks") or []
    out.setdefault("project", it["id"] if it else "")
    out.setdefault("done_when", it["done_when"] if it else "")
    out.setdefault("done", [t["title"] for t in tasks if t["status"] == "done"][-REPORT_LIST_MAX:])
    out.setdefault("left", [t["title"] for t in tasks if t["status"] in TASK_OPEN][:REPORT_LIST_MAX])
    return out
