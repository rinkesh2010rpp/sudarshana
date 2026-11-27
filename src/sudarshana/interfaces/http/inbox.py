"""The visitor inbox's public API (/api/inbox) as a plain FastAPI app."""
from sudarshana.memory.inbox import (
    INBOX_MAX_ITEM_LEN,
    INBOX_MAX_NAME_LEN,
    INBOX_PUBLIC_STATUSES,
    _InboxDB,
    _inbox_set_status,
    _inbox_ts,
)
from sudarshana.safety.jev_triage import (
    JEV_CONFIDENCE,
    JEV_ENABLED,
    JEV_VERDICTS,
    _jev_input_hash,
    _jev_triage,
)
def create_inbox_app(deps):
    """Visitor inbox endpoint.

    POST /api/inbox stores a submission (text + optional name) as 'received'
    (PRIVATE) until it passes the policy check. GET /api/inbox serves the
    public board. CORS is enabled so the gateway SPA can call it cross-origin.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uuid as _uuid

    web_app = FastAPI(title="sudarshana-inbox")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    class InboxSubmission(BaseModel):
        text: str = Field(..., min_length=1, max_length=INBOX_MAX_ITEM_LEN)
        name: str = Field(default="", max_length=INBOX_MAX_NAME_LEN)

    @web_app.post("/api/inbox")
    def submit(sub: InboxSubmission):
        text = sub.text.strip()
        name = sub.name.strip() or "anonymous"
        if not text:
            return {"ok": False, "error": "empty-item"}
        db = _InboxDB()
        ts = _inbox_ts()
        item_id = f"item:{_uuid.uuid4().hex[:12]}"
        rec = {
            "id": item_id,
            "name": name,
            "text": text,
            "status": "received",
            "created_at": ts,
            "updated_at": ts,
            "artifact_link": None,
        }
        db.insert(rec)
        db.log_event(
            item_id,
            from_status="-",
            to_status="received",
            note=f"name={name}",
            artifact_link="",
        )
        # Explicit commit: the container is torn down before auto-flush.
        deps.storage.commit()

        # Apply a confident Jev verdict now via _inbox_set_status; anything
        # else leaves the item 'received' for the agent's next-cycle review.
        triage = None
        outcome = None  # set only when a transition actually landed
        if JEV_ENABLED:
            triage = _jev_triage(text)
            verdict, confidence = triage.get("verdict"), triage.get("confidence", 0.0)
            if triage.get("ran") and verdict and confidence >= JEV_CONFIDENCE and verdict in JEV_VERDICTS:
                target = JEV_VERDICTS[verdict]
                applied = _inbox_set_status(
                    deps,
                    item_id,
                    target,
                    note=(
                        f"Jev triager: verdict={verdict} confidence={confidence:.2f} "
                        f"probs={triage.get('probabilities')} input_hash={_jev_input_hash(text)}"
                    ),
                )
                if not applied.get("ok"):
                    # Didn't land: log it and fall through to the 'received'
                    # response, so the visitor isn't told otherwise.
                    db.log_event(
                        item_id,
                        from_status="received",
                        to_status="received",
                        note=(
                            f"Jev triager: transition to {target} FAILED "
                            f"({applied.get('error') or 'unknown'}) — leaving "
                            f"received for manual review. "
                            f"input_hash={_jev_input_hash(text)}"
                        ),
                        artifact_link="",
                    )
                    deps.storage.commit()
                else:
                    outcome = target
            else:
                # No auto-action: a 'hold' verdict, low confidence, or a Jev
                # failure. Logged so the reason is auditable.
                if triage.get("ran") and verdict == "hold":
                    reason = f"hold-verdict:{confidence:.2f}"
                elif triage.get("ran"):
                    reason = f"low-confidence:{confidence:.2f}"
                else:
                    reason = triage.get("error") or "unknown"
                db.log_event(
                    item_id,
                    from_status="received",
                    to_status="received",
                    note=f"Jev triager: no auto-action ({reason}) input_hash={_jev_input_hash(text)}",
                    artifact_link="",
                )
                deps.storage.commit()

        if outcome == "submitted":
            return {
                "ok": True,
                "id": item_id,
                "status": "submitted",
                "message": (
                    "Received and added to the public queue — it meets the "
                    "moderation policy. I'll get to it as soon as I can."
                ),
            }
        if outcome == "rejected":
            return {
                "ok": True,
                "id": item_id,
                "status": "rejected",
                "message": (
                    "Not accepted — thanks for writing. This queue only carries "
                    "requests that meet the moderation policy."
                ),
            }
        return {
            "ok": True,
            "id": item_id,
            "status": "received",
            "message": (
                "Received — I'll review this on my next run, and if it meets "
                "policy it'll be added to the queue automatically."
            ),
        }

    @web_app.get("/api/inbox")
    def read(limit: int = 50):
        db = _InboxDB()
        rows = db.fetch(INBOX_PUBLIC_STATUSES)
        rows.sort(key=lambda r: r["created_at"] or "", reverse=True)
        rows = rows[: max(1, min(int(limit), 100))]
        items = []
        for r in rows:
            ans = db.get_answer(r["id"]) if r["status"] == "completed" else None
            items.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "name": r["name"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "artifact_link": r["artifact_link"],
                    "answer": (
                        {
                            "id": ans["id"],
                            "body": ans["body"],
                            "created_at": ans["created_at"],
                        }
                        if ans
                        else None
                    ),
                }
            )
        return {"generated_at": _inbox_ts(), "items": items, "count": len(items)}

    return web_app
