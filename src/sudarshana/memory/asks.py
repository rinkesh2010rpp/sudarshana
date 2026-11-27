"""The ask list: what the agent needs from Rinkesh."""
import os

from sudarshana.core.config import VOLUME_PATH

# --- asks: what the agent needs from Rinkesh ----------------------------------
# Code owns this list so it can't go stale the way hand-kept lists did (PR #64
# sat in state.md's "his:" line, and PR #61 in asks.json, for days after each
# merged). ask_rinkesh sends one Telegram message and records the ask;
# resolve_ask closes it; an ask nobody re-confirms lapses by itself.
#
# Stored in /data/memory/asks.json, the file /control's overview already
# serves: "asks" holds OPEN asks only (the panel shows that list as-is), with
# the fields it reads (id, ask, detail, asked_at); "closed" keeps the record.
# Turns run one at a time in one container, so a plain file is safe here.
ASKS_PATH = os.path.join(VOLUME_PATH, "memory", "asks.json")
ASKS_WHAT = (
    "What Sudarshana needs from Rinkesh, served by /api/control/overview. "
    "Written only by the ask_rinkesh / resolve_ask tools — do not edit by hand."
)
ASK_DEFAULT_DAYS = 3
ASK_MAX_DAYS = 14
# Asking is meant to be rare; a cap forces closing or dropping old asks first.
ASK_MAX_OPEN = 3
# Closed asks kept for the record, newest last.
ASK_KEEP_CLOSED = 50
ASK_RESOLVE_STATUSES = {"answered", "dropped"}


def _ask_normalize(it: dict, status: str):
    """One stored ask in the shape the tools use, or None if unusable.
    Tolerates hand-written entries (only id/ask/detail/asked_at): they become
    open asks that lapse ASK_DEFAULT_DAYS after asked_at, like any other."""
    from datetime import timedelta

    if not isinstance(it, dict):
        return None
    ask = str(it.get("ask") or it.get("question") or "").strip()
    asked_at = str(it.get("asked_at") or it.get("created") or "").strip()
    try:
        asked = _parse_ts(asked_at)
        if asked.tzinfo is None:
            return None
    except (TypeError, ValueError):
        return None
    if not ask:
        return None
    a = dict(it)
    a["id"] = str(it.get("id") or "").strip() or f"old-{asked.strftime('%m%d%H%M')}"
    a["ask"] = ask
    a["asked_at"] = asked.isoformat()
    a["why"] = str(it.get("why") or it.get("detail") or "").strip()
    a["detail"] = str(it.get("detail") or a["why"]).strip()
    a["status"] = str(it.get("status") or status)
    try:
        _parse_ts(str(it.get("expires")))
    except (TypeError, ValueError):
        a["expires"] = (asked + timedelta(days=ASK_DEFAULT_DAYS)).isoformat()
    a.pop("question", None)
    a.pop("created", None)
    return a


def _asks_load() -> list:
    """All asks, open first. A missing file is an empty list; an unreadable or
    malformed one raises, because "no asks" and "the list is broken" must not
    look the same (the same rule /control's reader follows)."""
    import json

    try:
        with open(ASKS_PATH, encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        return []
    if not raw.strip():
        return []
    data = json.loads(raw)
    if isinstance(data, list):
        data = {"asks": data}
    if not isinstance(data, dict) or not isinstance(data.get("asks", []), list):
        raise ValueError("asks.json has no asks list")
    out = []
    for status, items in (("open", data.get("asks") or []), ("closed", data.get("closed") or [])):
        for it in items if isinstance(items, list) else []:
            a = _ask_normalize(it, status)
            if a is not None:
                out.append(a)
    return out


def _asks_save(deps, asks: list) -> None:
    """Write atomically, so a crash mid-write can't leave /control a torn file,
    then commit the Volume at once. Waiting for the turn-end commit loses the
    record if the turn dies after the Telegram message went out (timeout,
    crash): he has the ask, the panel never shows it, and the next turn asks
    again. Same reason _inbox_set_status commits mid-turn."""
    import json

    open_ = [a for a in asks if a["status"] == "open"]
    closed = [a for a in asks if a["status"] != "open"][-ASK_KEEP_CLOSED:]
    os.makedirs(os.path.dirname(ASKS_PATH), exist_ok=True)
    tmp = ASKS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"_what": ASKS_WHAT, "asks": open_, "closed": closed}, f, indent=1, ensure_ascii=False)
    os.replace(tmp, ASKS_PATH)
    try:
        deps.storage.commit()
    except Exception as e:  # noqa: BLE001 — the turn-end commit still carries it
        print(f"[asks] volume commit failed, relying on the turn-end commit: {e!r}")


def _utcnow():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _parse_ts(s: str):
    from datetime import datetime

    return datetime.fromisoformat(s)


def _age(delta) -> str:
    secs = int(delta.total_seconds())
    if secs < 3600:
        return f"{max(secs // 60, 0)}m"
    if secs < 86400:
        return f"{secs // 3600}h"
    return f"{secs // 86400}d"


def _asks_expire(asks: list, now) -> list:
    """Mark open asks past their expiry as expired, in place. Returns them."""
    expired = []
    for a in asks:
        if a["status"] == "open" and _parse_ts(a["expires"]) <= now:
            a["status"] = "expired"
            a["closed"] = now.isoformat()
            a["outcome"] = "expired with no answer or re-confirmation"
            a["expiry_reported"] = False
            expired.append(a)
    return expired


def _ask_create(deps, question: str, why: str, options: str = "", expires_days: int = ASK_DEFAULT_DAYS,
                now=None) -> dict:
    """Record a new ask and send it to Rinkesh. Nothing is stored unless the
    message actually went out, so the list never shows an ask he never saw."""
    import secrets
    from datetime import timedelta

    question, why, options = (question or "").strip(), (why or "").strip(), (options or "").strip()
    if not question:
        return {"ok": False, "error": "question is empty"}
    if not why:
        return {"ok": False, "error": "say why you need this from him (why=...)"}
    try:
        days = int(expires_days)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"expires_days must be a whole number, got {expires_days!r}"}
    days = min(max(days, 1), ASK_MAX_DAYS)

    now = now or _utcnow()
    try:
        asks = _asks_load()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"can't read {ASKS_PATH} ({e}); nothing sent — fix the file first"}
    _asks_expire(asks, now)
    open_ = [a for a in asks if a["status"] == "open"]
    for a in open_:
        if a["ask"].strip().lower() == question.lower():
            return {"ok": False, "error": f"already open as [{a['id']}] — don't ask twice"}
    if len(open_) >= ASK_MAX_OPEN:
        ids = ", ".join(f"[{a['id']}]" for a in open_)
        return {
            "ok": False,
            "error": f"{len(open_)} asks already open ({ids}), the cap is {ASK_MAX_OPEN}. "
            "Resolve or drop one with resolve_ask first.",
        }

    taken = {a["id"] for a in asks}
    ask_id = secrets.token_hex(2)
    while ask_id in taken:
        ask_id = secrets.token_hex(2)

    text = f"Sudarshana needs you [ask {ask_id}]\n\n{question}\n\nWhy: {why}"
    if options:
        text += f"\nOptions: {options}"
    text += f"\n\nReply here. If it's still unanswered in {days} day(s), it lapses unless I re-ask."
    try:
        deps.notifier.send(text)
    except Exception as e:  # noqa: BLE001 — report, don't crash the turn
        return {"ok": False, "error": f"Telegram send failed, ask not recorded: {e}"}

    record = {
        "id": ask_id,
        "ask": question,
        # /control shows `detail` under the ask.
        "detail": why + (f" Options: {options}" if options else ""),
        "why": why,
        "options": options,
        "status": "open",
        "asked_at": now.isoformat(),
        "expires": (now + timedelta(days=days)).isoformat(),
    }
    asks.append(record)
    _asks_save(deps, asks)
    return {"ok": True, "id": ask_id, "expires": record["expires"]}


def _ask_resolve(deps, ask_id: str, outcome: str, status: str = "answered", now=None) -> dict:
    """Close one open ask with what came of it."""
    ask_id = (ask_id or "").strip().strip("[]").removeprefix("ask ").strip()
    outcome = (outcome or "").strip()
    if status not in ASK_RESOLVE_STATUSES:
        return {"ok": False, "error": f"status must be one of {sorted(ASK_RESOLVE_STATUSES)}"}
    if not outcome:
        return {"ok": False, "error": "say what came of it (outcome=...)"}
    now = now or _utcnow()
    try:
        asks = _asks_load()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"can't read {ASKS_PATH} ({e})"}
    _asks_expire(asks, now)
    for a in asks:
        if a["id"] != ask_id:
            continue
        if a["status"] != "open":
            return {"ok": False, "error": f"[{ask_id}] is already {a['status']}"}
        a["status"] = status
        a["outcome"] = outcome
        a["closed"] = now.isoformat()
        _asks_save(deps, asks)
        return {"ok": True, "id": ask_id, "status": status}
    return {"ok": False, "error": f"no ask with id [{ask_id}]"}


def _asks_context(deps, now=None) -> str:
    """Per-call note listing open asks and any that just lapsed. Empty when
    there are none, so a quiet cycle costs nothing. Each lapsed ask is
    reported once. Fails open: a store error must not cost the turn."""
    try:
        return _asks_context_unsafe(deps, now)
    except Exception as e:  # noqa: BLE001
        print(f"[asks] context unavailable: {e!r}")
        return "(Open asks to Rinkesh: the list could not be read this turn.)"


def _asks_context_unsafe(deps, now=None) -> str:
    now = now or _utcnow()
    asks = _asks_load()
    _asks_expire(asks, now)
    open_ = [a for a in asks if a["status"] == "open"]
    lapsed = [a for a in asks if a["status"] == "expired" and not a.get("expiry_reported", True)]
    # Anything that expired just now is in `lapsed`, so with none there is
    # nothing new to write — and no reason to commit the Volume every turn.
    if not open_ and not lapsed:
        return ""
    lines = []
    if open_:
        lines.append(
            f"Open asks to Rinkesh ({len(open_)}/{ASK_MAX_OPEN}). He has each one on "
            "Telegram already; don't repeat them in your messages. This is the whole "
            "list of what you need from him."
        )
        for a in open_:
            age = _age(now - _parse_ts(a["asked_at"]))
            left = _age(_parse_ts(a["expires"]) - now)
            lines.append(f"  [{a['id']}] asked {age} ago, lapses in {left}: {a['ask']}")
        lines.append(
            "If his message answers one, or the need has gone away, close it: "
            "resolve_ask(ask_id, outcome, status='answered'|'dropped')."
        )
    if lapsed:
        lines.append("Asks that lapsed with no answer (now closed):")
        for a in lapsed:
            lines.append(f"  [{a['id']}] {a['ask']}")
            a["expiry_reported"] = True
        lines.append(
            "If one is still truly blocking, ask_rinkesh again and say what changed; "
            "otherwise let it go."
        )
        _asks_save(deps, asks)  # the expiry and its reported-once mark
    return "\n".join(lines)
