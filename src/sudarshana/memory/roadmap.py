"""The roadmap: initiatives and their tasks, owned by code."""
import os

from sudarshana.core.config import VOLUME_PATH
from sudarshana.infra.storage.modal.volume import volume
from sudarshana.memory.asks import _utcnow

# --- roadmap: initiatives and their tasks, owned by code ----------------------
# ROADMAP.md grew to ~1,100 lines of prose where an initiative's status was the
# word in its heading ("Active", "Declined"), and /control could only serve the
# file raw. Every cycle also began by reading it to decide what to work on.
# So the structure moves into a store only the roadmap tools write:
#   roadmap -> initiatives (objective, done_when, status, doc)
#           -> tasks (title, done_when, status, handoff, deliverables)
# The prose stays in each initiative's doc, /data/actions/<id>.md, which the
# record links to. The same records feed the per-turn "Roadmap" note (what to
# work on) and /api/control/roadmap (what Rinkesh sees), so the two can't
# drift. Turns run one at a time in one container, so a plain file is safe,
# as with asks.json.
ROADMAP_PATH = os.path.join(VOLUME_PATH, "memory", "roadmap.json")
ACTIONS_DIR = os.path.join(VOLUME_PATH, "actions")
ACTIONS_ARCHIVE_DIR = os.path.join(ACTIONS_DIR, "archive")
ROADMAP_WHAT = (
    "Sudarshana's roadmap: initiatives and their tasks, served by "
    "/api/control/roadmap. Written only by the roadmap tools — do not edit by hand."
)
INITIATIVE_STATUSES = ("proposed", "active", "waiting", "blocked", "done", "declined")
INITIATIVE_OPEN = ("proposed", "active", "waiting", "blocked")
# Allowed moves. Closed states are final: reopening means a new initiative.
INITIATIVE_MOVES = {
    "proposed": {"active", "declined"},
    "active": {"waiting", "blocked", "done", "declined"},
    "waiting": {"active", "blocked", "declined"},
    "blocked": {"active", "declined"},
    "done": set(),
    "declined": set(),
}
TASK_STATUSES = ("todo", "doing", "waiting", "blocked", "done", "dropped")
TASK_OPEN = ("todo", "doing", "waiting", "blocked")
TASK_MOVES = {
    "todo": {"doing", "dropped"},
    "doing": {"todo", "waiting", "blocked", "done", "dropped"},
    "waiting": {"doing", "blocked", "dropped"},
    "blocked": {"doing", "dropped"},
    "done": set(),
    "dropped": set(),
}
ROADMAP_TEXT_MAX = 400
ROADMAP_HANDOFF_MAX = 600
ROADMAP_LIST_MAX = 20
ROADMAP_TASKS_MAX = 40
ROADMAP_ID_RE = r"[a-z0-9][a-z0-9-]{0,59}"


def _rm_text(s, cap: int = ROADMAP_TEXT_MAX) -> str:
    return " ".join(str(s or "").split())[:cap]


def _roadmap_load() -> list:
    """All initiatives. A missing file is an empty roadmap; an unreadable or
    malformed one raises, so "no initiatives" and "the store is broken" can't
    look the same (the rule asks.json follows)."""
    import json

    try:
        with open(ROADMAP_PATH, encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        return []
    if not raw.strip():
        return []
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("initiatives"), list):
        raise ValueError("roadmap.json has no initiatives list")
    return [it for it in data["initiatives"] if isinstance(it, dict) and it.get("id")]


def _roadmap_save(items: list) -> None:
    """Atomic write, then commit the Volume at once (a turn that dies later
    must not lose a status change it already reported)."""
    import json

    os.makedirs(os.path.dirname(ROADMAP_PATH), exist_ok=True)
    tmp = ROADMAP_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"_what": ROADMAP_WHAT, "initiatives": items}, f, indent=1, ensure_ascii=False)
    os.replace(tmp, ROADMAP_PATH)
    try:
        volume.commit()
    except Exception as e:  # noqa: BLE001 — the turn-end commit still carries it
        print(f"[roadmap] volume commit failed, relying on the turn-end commit: {e!r}")


def _rm_sorted(items: list) -> list:
    """Roadmap order: open before closed, then priority (lower first), then age."""
    return sorted(
        items,
        key=lambda i: (
            i.get("status") not in INITIATIVE_OPEN,
            int(i.get("priority") or 0),
            str(i.get("created") or ""),
        ),
    )


def _rm_find(items: list, initiative_id: str):
    initiative_id = str(initiative_id or "").strip().removesuffix(".md")
    for it in items:
        if it["id"] == initiative_id:
            return it
    return None


def _rm_task(it: dict, task_id: str):
    task_id = str(task_id or "").strip()
    for t in it.get("tasks") or []:
        if t["id"] == task_id:
            return t
    return None


def _rm_check_blocked_on(blocked_on: str):
    """None if fine, else an error. Free text: what it waits on ("ask 3f2a",
    "Netlify outage"). The agent keeps it current itself."""
    if not blocked_on:
        return "say what it waits on (blocked_on=...), e.g. 'ask 3f2a'"
    return None


def _rm_check_paths(paths, field: str):
    """Clean list of links/deliverables, or (None, error). /data paths must
    exist; URLs are taken as given."""
    out = []
    for p in paths or []:
        p = str(p or "").strip()
        if not p:
            continue
        if p.startswith(("http://", "https://")):
            out.append(p[:ROADMAP_TEXT_MAX])
            continue
        if not p.startswith(VOLUME_PATH + "/") or ".." in p.split("/"):
            return None, f"{field}: {p!r} is neither a URL nor a {VOLUME_PATH}/ path"
        if not os.path.exists(p):
            return None, f"{field}: {p} does not exist"
        out.append(p)
    return out[:ROADMAP_LIST_MAX], None


def _rm_doc_path(initiative_id: str) -> str:
    # /data paths are POSIX; built by hand so they match what the tools check.
    return f"{ACTIONS_DIR}/{initiative_id}.md"


def _rm_ensure_doc(it: dict) -> None:
    """Create the initiative's doc if it doesn't exist yet. An existing file
    (an initiative from before this store) is kept as it is."""
    path = it["doc"]
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# {it['id']} — {it['title']}\n\n"
            f"**Objective.** {it['objective']}\n\n"
            f"**Done when.** {it['done_when']}\n\n"
            "Status and tasks live in the roadmap store (roadmap_view); this file "
            "holds the notes, reasoning and findings.\n\n## Notes\n"
        )


def _rm_archive_doc(it: dict, now) -> None:
    """Move a closed initiative's doc to actions/archive/ and repoint `doc`,
    so the link stays valid. Never overwrites an existing archive file."""
    src = it.get("doc") or ""
    if not src or not os.path.isfile(src) or os.path.dirname(src) == ACTIONS_ARCHIVE_DIR:
        return
    os.makedirs(ACTIONS_ARCHIVE_DIR, exist_ok=True)
    dst = f"{ACTIONS_ARCHIVE_DIR}/{os.path.basename(src)}"
    if os.path.exists(dst):
        dst = f"{ACTIONS_ARCHIVE_DIR}/{it['id']}-{now.strftime('%Y-%m-%d')}.md"
        n = 2
        while os.path.exists(dst):
            dst = f"{ACTIONS_ARCHIVE_DIR}/{it['id']}-{now.strftime('%Y-%m-%d')}-{n}.md"
            n += 1
    os.replace(src, dst)
    it["doc"] = dst


def _initiative_create(initiative_id: str, title: str, objective: str, done_when: str,
                       asked_by: str = "", status: str = "proposed", priority: int = 0,
                       links=None, now=None) -> dict:
    import re

    now = now or _utcnow()
    initiative_id = str(initiative_id or "").strip().removesuffix(".md")
    if not re.fullmatch(ROADMAP_ID_RE, initiative_id):
        return {"ok": False, "error": "id must be short lower-kebab-case, e.g. gate-axe"}
    title, objective, done_when = _rm_text(title), _rm_text(objective), _rm_text(done_when)
    asked_by = _rm_text(asked_by)
    if not title or not objective or not done_when:
        return {"ok": False, "error": "title, objective and done_when are all required"}
    if status not in ("proposed", "active"):
        return {"ok": False, "error": "a new initiative starts as 'proposed' or 'active'"}
    try:
        prio = int(priority or 0)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"priority must be a whole number, got {priority!r}"}
    links, err = _rm_check_paths(links, "links")
    if err:
        return {"ok": False, "error": err}
    try:
        items = _roadmap_load()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"can't read {ROADMAP_PATH} ({e}); nothing changed"}
    if _rm_find(items, initiative_id):
        return {"ok": False, "error": f"initiative {initiative_id} already exists"}
    ts = now.isoformat()
    it = {
        "id": initiative_id,
        "title": title,
        "objective": objective,
        "done_when": done_when,
        "asked_by": asked_by,
        "status": status,
        "priority": prio,
        "doc": _rm_doc_path(initiative_id),
        "links": links,
        "blocked_on": "",
        "blocked_reason": "",
        "created": ts,
        "updated": ts,
        "closed": None,
        "outcome": "",
        "tasks": [],
    }
    _rm_ensure_doc(it)
    items.append(it)
    _roadmap_save(items)
    return {"ok": True, "initiative": it}


def _initiative_update(initiative_id: str, status: str = "", title: str = "", objective: str = "",
                       done_when: str = "", priority=None, blocked_on: str = "", reason: str = "",
                       outcome: str = "", links=None, now=None) -> dict:
    now = now or _utcnow()
    try:
        items = _roadmap_load()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"can't read {ROADMAP_PATH} ({e}); nothing changed"}
    it = _rm_find(items, initiative_id)
    if not it:
        return {"ok": False, "error": f"no initiative {initiative_id!r}"}
    if it["status"] not in INITIATIVE_OPEN:
        return {"ok": False, "error": f"{it['id']} is {it['status']} — closed initiatives don't change"}
    reason, outcome, blocked_on = _rm_text(reason), _rm_text(outcome), _rm_text(blocked_on)

    if priority is not None:
        try:
            priority = int(priority)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"priority must be a whole number, got {priority!r}"}
    if links is not None:
        links, err = _rm_check_paths(links, "links")
        if err:
            return {"ok": False, "error": err}

    status = str(status or "").strip()
    if status and status != it["status"]:
        if status not in INITIATIVE_STATUSES:
            return {"ok": False, "error": f"status must be one of {list(INITIATIVE_STATUSES)}"}
        if status not in INITIATIVE_MOVES[it["status"]]:
            allowed = sorted(INITIATIVE_MOVES[it["status"]])
            return {"ok": False, "error": f"{it['id']} can't go {it['status']} -> {status}; allowed: {allowed}"}
        if status == "waiting":
            err = _rm_check_blocked_on(blocked_on)
            if err:
                return {"ok": False, "error": err}
        if status == "blocked" and not reason:
            return {"ok": False, "error": "say why it's blocked (reason=...)"}
        if status in ("done", "declined") and not outcome:
            return {"ok": False, "error": "say what came of it (outcome=...)"}

    # All checks passed — apply.
    for field, val in (("title", title), ("objective", objective), ("done_when", done_when)):
        if _rm_text(val):
            it[field] = _rm_text(val)
    if priority is not None:
        it["priority"] = priority
    if links is not None:
        it["links"] = links
    if status and status != it["status"]:
        it["status"] = status
        it["blocked_on"] = blocked_on if status == "waiting" else ""
        it["blocked_reason"] = reason if status == "blocked" else ""
        if status in ("done", "declined"):
            it["closed"] = now.isoformat()
            it["outcome"] = outcome
            _rm_archive_doc(it, now)
    it["updated"] = now.isoformat()
    _roadmap_save(items)
    return {"ok": True, "initiative": it}


def _task_add(initiative_id: str, title: str, done_when: str, position=None, now=None) -> dict:
    now = now or _utcnow()
    title, done_when = _rm_text(title), _rm_text(done_when)
    if not title or not done_when:
        return {"ok": False, "error": "title and done_when are both required"}
    try:
        items = _roadmap_load()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"can't read {ROADMAP_PATH} ({e}); nothing changed"}
    it = _rm_find(items, initiative_id)
    if not it:
        return {"ok": False, "error": f"no initiative {initiative_id!r}"}
    if it["status"] not in INITIATIVE_OPEN:
        return {"ok": False, "error": f"{it['id']} is {it['status']} — closed initiatives don't change"}
    tasks = it.setdefault("tasks", [])
    if sum(1 for t in tasks if t["status"] in TASK_OPEN) >= ROADMAP_TASKS_MAX:
        return {"ok": False, "error": f"{it['id']} already has {ROADMAP_TASKS_MAX} open tasks — finish or drop some"}
    n = 1 + max((int(t["id"][1:]) for t in tasks if t["id"][1:].isdigit()), default=0)
    ts = now.isoformat()
    task = {
        "id": f"t{n}",
        "title": title,
        "done_when": done_when,
        "status": "todo",
        "handoff": "",
        "deliverables": [],
        "blocked_on": "",
        "blocked_reason": "",
        "created": ts,
        "updated": ts,
        "closed": None,
    }
    try:
        pos = None if position is None or position == "" else int(position)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"position must be a whole number, got {position!r}"}
    if pos is None:
        tasks.append(task)
    else:
        tasks.insert(max(pos - 1, 0), task)
    it["updated"] = ts
    _roadmap_save(items)
    return {"ok": True, "task": task, "initiative": it["id"]}


def _task_update(initiative_id: str, task_id: str, status: str = "", title: str = "",
                 done_when: str = "", handoff: str = "", deliverables=None,
                 blocked_on: str = "", reason: str = "", now=None) -> dict:
    now = now or _utcnow()
    try:
        items = _roadmap_load()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"can't read {ROADMAP_PATH} ({e}); nothing changed"}
    it = _rm_find(items, initiative_id)
    if not it:
        return {"ok": False, "error": f"no initiative {initiative_id!r}"}
    task = _rm_task(it, task_id)
    if not task:
        return {"ok": False, "error": f"{it['id']} has no task {task_id!r}"}
    if task["status"] not in TASK_OPEN:
        return {"ok": False, "error": f"{it['id']}/{task['id']} is {task['status']} — closed tasks don't change"}
    blocked_on, reason = _rm_text(blocked_on), _rm_text(reason)
    new_deliverables, err = _rm_check_paths(deliverables, "deliverables")
    if err:
        return {"ok": False, "error": err}

    status = str(status or "").strip()
    if status and status != task["status"]:
        if status not in TASK_STATUSES:
            return {"ok": False, "error": f"status must be one of {list(TASK_STATUSES)}"}
        if status not in TASK_MOVES[task["status"]]:
            allowed = sorted(TASK_MOVES[task["status"]])
            return {"ok": False, "error": f"{task['id']} can't go {task['status']} -> {status}; allowed: {allowed}"}
        if status == "doing":
            if it["status"] != "active":
                return {"ok": False, "error": f"{it['id']} is {it['status']} — tasks start only on an active initiative"}
            busy = [t["id"] for t in it["tasks"] if t["status"] == "doing" and t is not task]
            if busy:
                return {"ok": False, "error": f"{busy[0]} is already doing — one task at a time; move it first"}
        if status == "waiting":
            err = _rm_check_blocked_on(blocked_on)
            if err:
                return {"ok": False, "error": err}
        if status == "blocked" and not reason:
            return {"ok": False, "error": "say why it's blocked (reason=...)"}
        if status == "done" and not (new_deliverables or task["deliverables"]):
            return {"ok": False, "error": "a done task names what it produced: deliverables=[URL or /data path]"}
        if status in ("done", "dropped") and not (_rm_text(handoff) or reason):
            return {"ok": False, "error": "say what came of it (handoff= for done, reason= for dropped)"}

    for field, val in (("title", title), ("done_when", done_when)):
        if _rm_text(val):
            task[field] = _rm_text(val)
    if _rm_text(handoff, ROADMAP_HANDOFF_MAX):
        task["handoff"] = _rm_text(handoff, ROADMAP_HANDOFF_MAX)
    for d in new_deliverables:
        if d not in task["deliverables"]:
            task["deliverables"].append(d)
    task["deliverables"] = task["deliverables"][-ROADMAP_LIST_MAX:]
    if status and status != task["status"]:
        task["status"] = status
        task["blocked_on"] = blocked_on if status == "waiting" else ""
        task["blocked_reason"] = reason if status in ("blocked", "dropped") else ""
        if status in ("done", "dropped"):
            task["closed"] = now.isoformat()
    task["updated"] = it["updated"] = now.isoformat()
    _roadmap_save(items)
    return {"ok": True, "task": task, "initiative": it["id"]}


def _roadmap_focus(items: list):
    """The initiative this cycle should work on: the first active one in
    roadmap order (waiting/blocked ones are skipped), or None."""
    for it in _rm_sorted(items):
        if it["status"] == "active":
            return it
    return None


def _rm_task_line(t: dict) -> str:
    line = f"  {t['id']} [{t['status']}] {t['title']} — done when: {t['done_when']}"
    if t.get("blocked_on"):
        line += f" (waiting on {t['blocked_on']})"
    if t.get("blocked_reason") and t["status"] == "blocked":
        line += f" (blocked: {t['blocked_reason']})"
    return line


def _roadmap_context(now=None) -> str:
    """Per-turn note: the focus initiative with its current task and handoff,
    then one line for every other open initiative. Fails open."""
    try:
        return _roadmap_context_unsafe(now)
    except Exception as e:  # noqa: BLE001
        print(f"[roadmap] context unavailable: {e!r}")
        return "(Roadmap: the store could not be read this turn — check /data/memory/roadmap.json.)"


def _roadmap_context_unsafe(now=None) -> str:
    items = _roadmap_load()
    if not items:
        return (
            "Roadmap: the store is empty. Until it holds your initiatives, "
            "/data/ROADMAP.md is still where they are — migrate them (see 'The roadmap' "
            "in your instructions) as this cycle's work."
        )
    open_ = [i for i in _rm_sorted(items) if i["status"] in INITIATIVE_OPEN]
    if not open_:
        return "Roadmap: no open initiatives. Propose one (create_initiative) and put it to Rinkesh."
    lines = ["Roadmap (from the roadmap store; roadmap_view for detail):"]
    focus = _roadmap_focus(items)
    if focus:
        lines.append(f"FOCUS — {focus['id']}: {focus['title']} (doc {focus['doc']})")
        lines.append(f"  Objective: {focus['objective']}")
        lines.append(f"  Done when: {focus['done_when']}")
        tasks = [t for t in focus.get("tasks") or [] if t["status"] in TASK_OPEN]
        cur = next((t for t in tasks if t["status"] == "doing"), None) or \
            next((t for t in tasks if t["status"] == "todo"), None)
        if cur:
            lines.append(f"  Current task: {_rm_task_line(cur).strip()}")
            if cur.get("handoff"):
                lines.append(f"  Handoff from last cycle: {cur['handoff']}")
        else:
            lines.append("  No open tasks: add the next ones (add_task) or close it if done_when is met.")
        rest = [t for t in tasks if t is not cur][:4]
        if rest:
            lines.append("  Queue:")
            lines.extend(_rm_task_line(t) for t in rest)
    for it in open_:
        if it is focus:
            continue
        line = f"- {it['id']} [{it['status']}] {it['title']}"
        if it["status"] == "waiting":
            line += f" — waiting on {it['blocked_on']}"
        elif it["status"] == "blocked":
            line += f" — blocked: {it.get('blocked_reason') or '?'}"
        elif it["status"] == "proposed":
            line += " — proposed, not yet approved by Rinkesh: light research only"
        lines.append(line)
    if not focus:
        lines.append("No active initiative to work on: every open one is waiting, blocked or proposed.")
    lines.append(
        "Anything waiting on an ask: if the asks note shows it answered or lapsed, "
        "move it on yourself (update_task / update_initiative)."
    )
    return "\n".join(lines)


def _roadmap_view(initiative_id: str = "") -> str:
    """Text view for the model: the whole roadmap, or one initiative in full."""
    try:
        items = _roadmap_load()
    except Exception as e:  # noqa: BLE001
        return f"can't read {ROADMAP_PATH}: {e}"
    if initiative_id:
        it = _rm_find(items, initiative_id)
        if not it:
            return f"no initiative {initiative_id!r}"
        lines = [
            f"{it['id']} [{it['status']}] {it['title']} (priority {it['priority']})",
            f"Objective: {it['objective']}",
            f"Done when: {it['done_when']}",
            f"Doc: {it['doc']}",
        ]
        if it.get("asked_by"):
            lines.append(f"Asked by: {it['asked_by']}")
        if it.get("blocked_on"):
            lines.append(f"Waiting on: {it['blocked_on']}")
        if it.get("blocked_reason"):
            lines.append(f"Blocked: {it['blocked_reason']}")
        if it.get("links"):
            lines.append("Links: " + ", ".join(it["links"]))
        if it.get("outcome"):
            lines.append(f"Outcome: {it['outcome']}")
        lines.append("Tasks:")
        for t in it.get("tasks") or []:
            lines.append(_rm_task_line(t))
            if t.get("handoff"):
                lines.append(f"      handoff: {t['handoff']}")
            if t.get("deliverables"):
                lines.append("      deliverables: " + ", ".join(t["deliverables"]))
        if not it.get("tasks"):
            lines.append("  (none)")
        return "\n".join(lines)
    if not items:
        return "The roadmap store is empty."
    lines = []
    for it in _rm_sorted(items):
        open_tasks = sum(1 for t in it.get("tasks") or [] if t["status"] in TASK_OPEN)
        lines.append(f"{it['id']} [{it['status']}] p{it['priority']} {it['title']} — {open_tasks} open task(s)")
    return "\n".join(lines)
