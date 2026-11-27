"""The visitor inbox store: SQLite on the workspace, intake note, status moves."""
import os

from sudarshana.core.config import VOLUME_PATH

# --- visitor inbox: public "put item in inbox" surface -----------------------
# Strangers submit items via the public site, stored in one SQLite file on the
# Volume (durable, no expiry). Status flow: received (PRIVATE) -> submitted
# (PUBLIC) -> in_progress -> completed; 'rejected' is a private terminal state.
# The public GET serves only INBOX_PUBLIC_STATUSES — that filter is the
# moderation enforcement.
INBOX_DB_PATH = os.path.join(VOLUME_PATH, "visitor-inbox.db")
INBOX_MAX_ITEM_LEN = 2000
INBOX_MAX_NAME_LEN = 100
INBOX_PUBLIC_STATUSES = {"submitted", "in_progress", "completed"}

def _inbox_ts() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _InboxDB:
    """SQLite store for the visitor inbox at /data/visitor-inbox.db. Each call
    reconnects so the latest data is always read.

    Callers must commit storage after writes: web-endpoint containers are torn
    down before the Volume's auto-flush, so uncommitted writes are lost."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS inbox_items (
        id            TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        text          TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT 'received',
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL,
        artifact_link TEXT
    );
    -- Task answers: long-form content entities of their own, paragraphs like a
    -- blog post but NOT blogs — kept apart from the blog collection and served
    -- on the item's detail view. The item row links here (answer_id), it does
    -- not carry the text.
    CREATE TABLE IF NOT EXISTS inbox_answers (
        id         TEXT PRIMARY KEY,
        item_id    TEXT NOT NULL,
        body       TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_inbox_answers_item ON inbox_answers(item_id);
    CREATE INDEX IF NOT EXISTS idx_inbox_status ON inbox_items(status);
    CREATE TABLE IF NOT EXISTS inbox_events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id      TEXT NOT NULL,
        from_status  TEXT NOT NULL,
        to_status    TEXT NOT NULL,
        note         TEXT,
        artifact_link TEXT,
        ts           TEXT NOT NULL
    );
    """

    def __init__(self, path: str = INBOX_DB_PATH):
        self.path = path
        conn = self._connect()
        try:
            conn.executescript(self.SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        # Rollback journal (not WAL): WAL's -shm/-wal sidecar files can break on
        # a network volume; DELETE journal + one writer lane is the safe shape.
        conn.execute("PRAGMA journal_mode=DELETE")
        return conn

    def insert(self, rec: dict):
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO inbox_items (id, name, text, status, created_at, updated_at, artifact_link)"
                " VALUES (:id, :name, :text, :status, :created_at, :updated_at, :artifact_link)",
                rec,
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, item_id: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM inbox_items WHERE id = ?", (item_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def fetch(self, statuses) -> list:
        """All items with status in the given set, oldest first."""
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(statuses))
            rows = conn.execute(
                f"SELECT * FROM inbox_items WHERE status IN ({placeholders})"
                " ORDER BY created_at",
                tuple(statuses),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def set_status(self, item_id: str, new_status: str, artifact_link: str = "") -> dict | None:
        """Transition one item; returns the updated row (with the pre-update
        status in `old_status`), or None if no such item. Raises ValueError on
        a forward-only violation. Persists artifact_link only when the status is
        'completed' (it is a public board artifact to the external result, or
        the task answer's own id, which has a long-form body in inbox_answers)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM inbox_items WHERE id = ?", (item_id,)
            ).fetchone()
            if not row:
                return None
            old = row["status"]
            if new_status not in INBOX_ALLOWED_TRANSITIONS.get(old, set()):
                raise ValueError(f"invalid transition {old} -> {new_status}")
            ts = _inbox_ts()
            if artifact_link and new_status == "completed":
                conn.execute(
                    "UPDATE inbox_items SET status = ?, updated_at = ?, artifact_link = ?"
                    " WHERE id = ?",
                    (new_status, ts, artifact_link, item_id),
                )
            else:
                conn.execute(
                    "UPDATE inbox_items SET status = ?, updated_at = ? WHERE id = ?",
                    (new_status, ts, item_id),
                )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM inbox_items WHERE id = ?", (item_id,)
            ).fetchone()
            updated = dict(row)
            updated["old_status"] = old
            return updated
        finally:
            conn.close()

    def put_answer(self, item_id: str, body: str) -> dict:
        """Create (or replace) the long-form answer content entity for an item.
        The item row carries only its id via artifact_link — the body lives
        here, served with the item on the detail view. Returns the answer row.
        NOTE: the caller (set_status path) commits the Volume after both the
        answer write and the item transition — see _inbox_set_status."""
        conn = self._connect()
        try:
            import uuid

            existing = conn.execute(
                "SELECT id FROM inbox_answers WHERE item_id = ?", (item_id,)
            ).fetchone()
            ts = _inbox_ts()
            answer_id = existing["id"] if existing else f"answer:{uuid.uuid4().hex[:12]}"
            if existing:
                conn.execute(
                    "UPDATE inbox_answers SET body = ?, updated_at = ? WHERE id = ?",
                    (body, ts, answer_id),
                )
            else:
                conn.execute(
                    "INSERT INTO inbox_answers (id, item_id, body, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (answer_id, item_id, body, ts, ts),
                )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM inbox_answers WHERE id = ?", (answer_id,)
            ).fetchone()
            return dict(row)
        finally:
            conn.close()

    def get_answer(self, item_id: str) -> dict | None:
        """The answer content entity for an item, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM inbox_answers WHERE item_id = ?", (item_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def log_event(self, item_id: str, from_status: str, to_status: str, note: str = "", artifact_link: str = ""):
        """Private audit trail of submissions + policy transitions (also where a
        model note goes now — the old mirror's job, inside the same store)."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO inbox_events (item_id, from_status, to_status, note, artifact_link, ts)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, from_status, to_status, note, artifact_link or None, _inbox_ts()),
            )
            conn.commit()
        finally:
            conn.close()


# Forward-only. 'rejected' is separate from 'completed' because completed is
# public; completed and rejected are terminal.
INBOX_ALLOWED_TRANSITIONS = {
    "received": {"submitted", "rejected"},
    "submitted": {"in_progress", "rejected"},
    "in_progress": {"completed"},
}


def _inbox_intake() -> dict:
    """Cycle-start view of the visitor inbox: PRIVATE received items (pending
    the policy check) and ACTIVE submitted/in_progress items (public,
    awaiting work)."""
    db = _InboxDB()
    pending = [
        {
            "id": r["id"],
            "name": r["name"] or "anonymous",
            "text": r["text"] or "",
            "created_at": r["created_at"] or "",
        }
        for r in db.fetch(["received"])
    ]
    active = [
        {
            "id": r["id"],
            "name": r["name"] or "anonymous",
            "text": r["text"] or "",
            "status": r["status"],
            "created_at": r["created_at"] or "",
        }
        for r in db.fetch(["submitted", "in_progress"])
    ]
    return {"pending": pending, "active": active}


def _inbox_intake_context() -> str:
    """Compact text of the intake view for the per-call system note. Empty
    string when the inbox is fully idle (the common case) so no-op cycles cost
    nothing."""
    view = _inbox_intake()
    lines = []
    if view["pending"]:
        lines.append(
            f"Visitor inbox: {len(view['pending'])} received item(s) awaiting your "
            "policy check. They are PRIVATE until you act."
        )
        for i, it in enumerate(view["pending"], 1):
            clip = it["text"][:200] + ("…" if len(it["text"]) > 200 else "")
            lines.append(f"  {i}. [{it['id']}] {it['name']}: {clip} (received {it['created_at']})")
        lines.append(
            "Policy check now: approve -> inbox_set_status '<id>' submitted (makes it "
            "PUBLIC on the site board); otherwise reject -> 'rejected' (never public). "
            "Never publish without this check."
        )
    if view["active"]:
        lines.append(
            f"Visitor inbox board: {len(view['active'])} item(s) public and awaiting work."
        )
        for i, it in enumerate(view["active"], 1):
            clip = it["text"][:120] + ("…" if len(it["text"]) > 120 else "")
            lines.append(f"  {i}. [{it['id']}] ({it['status']}) {it['name']}: {clip}")
        lines.append("If you have capacity, take one on: submitted -> in_progress -> completed.")
    return "\n".join(lines)


def _inbox_set_status(deps, item_id: str, new_status: str, note: str = "", artifact_link: str = "", answer: str = "") -> dict:
    """Apply one forward-only status transition to an inbox item. This is the
    ONLY way an item leaves the private 'received' set or is taken off the
    public board (submitted -> rejected is a retraction) — call it only after
    a policy judgment. `answer` is the full completed-item response, written
    as its OWN long-form content entity (inbox_answers): paragraphs like a
    blog post but NOT a blog — the item row carries only the link to it, and
    the detail view serves both. Returns a result dict {ok, error?, status?,
    public?}; public=True means the item is now served on the public board."""
    if new_status not in {"received", "submitted", "in_progress", "completed", "rejected"}:
        return {"ok": False, "error": f"unknown status {new_status!r}"}
    db = _InboxDB()
    try:
        if new_status == "completed" and answer:
            ans = db.put_answer(item_id, answer)
            artifact_link = ans["id"]
            row = db.set_status(item_id, new_status, artifact_link=artifact_link)
        else:
            row = db.set_status(item_id, new_status, artifact_link=artifact_link)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    if not row:
        return {"ok": False, "error": f"no such item {item_id!r}"}
    db.log_event(
        item_id,
        from_status=row["old_status"],
        to_status=new_status,
        note=note,
        artifact_link=artifact_link if new_status == "completed" else "",
    )
    # Without an explicit commit the transition (and answer) never reach the
    # shared Volume, so the public GET and next-cycle intake never see them.
    deps.storage.commit()
    return {
        "ok": True,
        "id": item_id,
        "status": new_status,
        "public": new_status in INBOX_PUBLIC_STATUSES,
    }
