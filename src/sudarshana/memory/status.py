"""Status center: deterministic turn-event writer on the shared status Dict."""
import time

import modal

# --- status-center: deterministic event writer + read-only API ---------------
# A named modal.Dict shared by the writer (agent containers) and the public,
# read-only API (mounts no volume). State:
#   * current       — idle / running-* + since
#   * current_trace — per-turn events, reset each turn
#   * history       — the last STATUS_HISTORY_LIMIT turn summaries (/api/events)
# plus per-turn records (turn:<ts>): one with no `ended` marks a turn that died
# mid-run. Unique per-turn keys make concurrent turns safe.
STATUS_DICT_NAME = "sudarshana-status"
STATUS_HISTORY_LIMIT = 50
# Closed turn:* records are pruned at turn-end by the timestamp in their key.
# The overview verdict covers 24 h, so 36 h of records is margin — and it is
# the record count that costs, not the store size: every read of this Dict is
# one gRPC message per entry (~8 ms warm), so 102 records made the panel's
# verdict ~1 s and every turn's bookkeeping ~9 s. At ~1.2 turns/hour, 36 h is
# ~44 records, which brings both down with no new code path and no new race.
STATUS_RECORDS_TTL_SECONDS = 36 * 3600
# Belt-and-braces cap for a burst of turns, applied oldest-first by key after
# the age prune. Never the primary rule: the panel needs 24 h, not N records.
STATUS_RECORDS_KEEP = 60
# An unclosed record older than this can't be a live turn — it's a zombie from a
# turn that died before _status_turn_end, and must not be handed `current`.
# This is the turn cap itself (record_timeout=1500), not a round 30 minutes: with
# 1800 a crashed turn read as "running" and healthy for the extra 5 minutes after
# the container could no longer possibly be alive.
STATUS_ZOMBIE_TTL_SECONDS = 1500


def _status_dict():
    """Handle to the shared status Dict (writer + API read the same one)."""
    return modal.Dict.from_name(STATUS_DICT_NAME, create_if_missing=True)


def _status_ts() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _status_guard(fn):
    """Run a status-write safely: on error, log loudly but never break the turn."""

    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            print(f"[status] write failed (fail-open): {type(e).__name__}: {e}")
            return None

    return wrapped


@_status_guard
def _status_turn_start(state: str) -> str:
    """Record that a turn of `state` (running-telegram/hourly/self-task) began.

    Writes a fresh immutable per-turn record (unique key -> safe under
    concurrent turns) and sets `current` + a fresh `current_trace`. Returns the
    record key, which the caller must hand back to `_status_turn_end` so each
    turn closes exactly its own record even when turns overlap. If a turn dies
    mid-run, `current` stays `running-*` and the record has no `ended` — a
    provable death.
    """
    import uuid

    ts = _status_ts()
    key = f"turn:{ts}"
    turn_id = f"{state}-{ts}-{uuid.uuid4().hex[:4]}"
    d = _status_dict()
    d.put(
        key,
        {
            "turn_id": turn_id,
            "state": state,
            "started": ts,
            "ended": None,
            "outcome": "running",
            "summary": "",
        },
        skip_if_exists=True,
    )
    d["current"] = {"state": state, "since": ts, "turn_id": turn_id, "_turn_key": key}
    d["current_trace"] = [{"ts": ts, "event": "turn_started", "detail": state}]
    _status_reconcile_zombies(d)
    return key


@_status_guard
def _status_parse_ts(iso: str):
    """Parse an ISO-8601 status timestamp to a UTC epoch float. Returns None on
    any malformed/absent value so callers can treat it as stale rather than
    crash."""
    if not iso:
        return None
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:  # noqa: BLE001
        return None


def _status_is_zombie(rec) -> bool:
    """True if a turn record is dead-but-unclosed: no `ended`, and started so
    long ago that no live turn (capped by the Modal timeout) could still be
    running. Such a record is a mid-run death that never got _status_turn_end."""
    if rec.get("ended") is not None:
        return False
    started = _status_parse_ts(rec.get("started"))
    if started is None:
        return False  # can't prove staleness; leave it
    now = time.time()
    return (now - started) > STATUS_ZOMBIE_TTL_SECONDS


@_status_guard
def _status_reconcile_zombies(d):
    """Close any ended=None record older than the staleness TTL, stamping it
    outcome="zombie" so the death is recorded (the turn:* records exist
    precisely to catch mid-run deaths) and so it is no longer mistaken for a
    live concurrent turn. Returns a list of (key, rec) for the zombies closed."""
    closed = []
    now = _status_ts()
    for k in d.keys() or []:
        if not (isinstance(k, str) and k.startswith("turn:")):
            continue
        rec = d.get(k, {}) or {}
        if _status_is_zombie(rec):
            rec["ended"] = rec.get("ended") or now
            # Unconditional, and deliberately so: a record with `ended` set is
            # never a zombie, so no honest outcome can be overwritten here.
            # The old `rec.get("outcome") or "zombie"` could never fire —
            # _status_turn_start writes outcome="running", which is truthy, so
            # a reconciled death kept saying "running" and the panel showed a
            # crashed newest turn as still running.
            rec["outcome"] = "zombie"
            rec["summary"] = rec.get("summary") or "mid-run death (stale > TTL); closed by reconciliation"
            d[k] = rec
            closed.append((k, rec))
    return closed


@_status_guard
def _status_prune(d, keep_key: str = ""):
    """Delete turn:* records older than STATUS_RECORDS_TTL_SECONDS, then cap
    what is left at STATUS_RECORDS_KEEP (oldest first).

    Decided entirely from the timestamp in each key, so the sweep costs one
    d.keys() and no per-record read — the old version asked every record
    "are you closed?" with a d.get(), i.e. one gRPC message per record on
    every turn end. Nothing unclosed is pruned silently: this runs after
    _status_reconcile_zombies in _status_turn_end, so a record past the TTL
    has already been closed as a death, and the running turn's own record is
    kept regardless."""
    keys = d.keys() or []
    now = time.time()
    turn_keys = sorted(
        k
        for k in keys
        if isinstance(k, str) and k.startswith("turn:") and k != keep_key
    )
    stale = []
    for k in turn_keys:
        started = _status_parse_ts(k[len("turn:"):])
        if started is None:
            continue
        if (now - started) > STATUS_RECORDS_TTL_SECONDS:
            stale.append(k)
    live = [k for k in turn_keys if k not in stale]
    excess = len(live) - STATUS_RECORDS_KEEP
    if excess > 0:
        # The cap reads only the few records it would delete — and only to keep
        # the one guarantee the age rule can't make on its own: an unclosed
        # record is a mid-run death, and deleting it would erase the evidence
        # (the old count-only prune never dropped one either). Under the cap
        # this branch does zero reads; it is a burst backstop, not the rule.
        for k in live[:excess]:
            if (d.get(k, {}) or {}).get("ended") is not None:
                stale.append(k)
    for k in stale:
        try:
            del d[k]
        except Exception as e:  # noqa: BLE001
            print(f"[status] prune step failed (non-fatal): {type(e).__name__}: {e}")
    return len(stale)


@_status_guard
def _status_other_running(d, exclude_key: str):
    """Most recent turn:* record (excluding exclude_key) that is genuinely still
    running: ended=None AND started within the staleness TTL. ISO-8601 keys sort
    lexicographically, so the max key is the newest start. Used to keep `current`
    honest when turns overlap (cron firing while a prior turn still runs): closing
    one turn must not idle a still-running one. Records older than the TTL are
    zombies and are never treated as running (they may be closed by
    _status_reconcile_zombies)."""
    best = None
    for k in d.keys() or []:
        if isinstance(k, str) and k.startswith("turn:") and k != exclude_key:
            rec = d.get(k, {}) or {}
            if rec.get("ended") is None and not _status_is_zombie(rec):
                if best is None or k > best[0]:
                    best = (k, rec)
    return best


@_status_guard
def _status_turn_end(turn_key: str, outcome: str, summary: str = ""):
    """Close the turn identified by `turn_key`: stamp ended/outcome on its
    record, prepend a compact summary to the rolling `history` (bounded to
    50), then — only if this turn still owns `current` — go idle, or hand
    `current` to another still-running turn if one exists (overlap case)."""
    d = _status_dict()
    ts = _status_ts()
    if not turn_key:
        return
    rec = d.get(turn_key, {}) or {}
    rec["ended"] = ts
    rec["outcome"] = outcome
    rec["summary"] = summary
    d[turn_key] = rec
    history = d.get("history", []) or []
    history.insert(
        0,
        {
            "turn_id": rec["turn_id"],
            "state": rec["state"],
            "started": rec["started"],
            "ended": rec["ended"],
            "outcome": rec["outcome"],
            "summary": rec["summary"],
        },
    )
    d["history"] = history[:STATUS_HISTORY_LIMIT]

    # Before deciding who gets `current`, so a zombie isn't taken for a live turn.
    _status_reconcile_zombies(d)

    # Prune records the overview no longer needs: by age, from the key alone.
    _status_prune(d, turn_key)

    current = d.get("current", {}) or {}
    if current.get("_turn_key") == turn_key:
        other = _status_other_running(d, turn_key)
        if other:
            okey, orec = other
            d["current"] = {
                "state": orec["state"],
                "since": orec["started"],
                "turn_id": orec["turn_id"],
                "_turn_key": okey,
            }
            # The other turn's trace was clobbered by this turn's start; mark
            # the handoff honestly rather than fabricating events.
            d["current_trace"] = [
                {"ts": _status_ts(), "event": "turn_resumed", "detail": "concurrent turn still running"}
            ]
        else:
            d["current"] = {"state": "idle", "since": ts}
            d["current_trace"] = []
