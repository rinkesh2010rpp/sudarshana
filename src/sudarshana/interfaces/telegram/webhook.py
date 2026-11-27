"""Telegram intake: the webhook's checks, dedup and spawn, with no Modal in it."""
import os

from sudarshana.core.dedup import is_seen, mark_seen

# Dedup state: a rolling window of recent update_ids per role, in one Dict
# entry each. "webhook" means the request was spawned; "run" means a turn was
# started. Separate roles so the two gates can't consume each other's claim.
TELEGRAM_CLAIM_DICT_NAME = "sudarshana-telegram-claims"
TELEGRAM_CLAIM_WINDOW = 200


def _telegram_seen(claims, update_id: int, role: str) -> bool:
    """True if `update_id` is already recorded for `role`. Advisory only.

    If the Dict can't be read this answers False — treat the update as new and
    print why. A claim that fails must never drop a real message; the cost of
    being wrong that way is one extra turn, the cost of the other way is a
    message that no retry can ever recover.

    Correctness against duplicates does not rest here. This read and the write
    in `_telegram_record` are two round trips, so two webhook containers can
    interleave and lose one id. The guarantee is the "run" gate in
    `process_message`, where calls are serialised by the class's single
    container and read-then-write cannot interleave with another copy of
    itself. This gate only saves a spawn.
    """
    try:
        return is_seen(claims, f"seen:{role}", update_id)
    except Exception as e:  # noqa: BLE001 — a Dict outage must not 500 the webhook
        print(f"[telegram] claim read failed (role={role}): {e!r} — treating as new")
        return False


def _telegram_record(claims, update_id: int, role: str) -> None:
    """Best-effort: remember `update_id` for `role` in one bounded list.

    Never raises — a Dict outage means a resend may re-run the update, which
    the "run" gate still stops from becoming a second turn *if it can read the
    Dict*, and which at worst costs one extra turn. Losing the message would be
    worse.
    """
    try:
        mark_seen(claims, f"seen:{role}", update_id, TELEGRAM_CLAIM_WINDOW)
    except Exception as e:  # noqa: BLE001
        print(f"[telegram] claim write failed (role={role}): {e!r} — continuing")


def handle_update(deps, payload: dict, secret_header: str, spawn) -> object:
    """Ack Telegram fast, then spawn the turn via `spawn(text, update_id)`. Never runs the model itself.

    Authenticity: the endpoint URL and the owner's numeric id are both
    discoverable (the URL is derivable from the workspace + label, the id is not
    secret), so the id check alone lets anyone who knows them POST a forged
    update that spawns a real turn against the agent's shell. Telegram stamps
    every genuine delivery with a shared secret in the
    X-Telegram-Bot-Api-Secret-Token header (registered via set_webhook.py); we
    reject anything whose header does not match *before* reading the body. See
    TELEGRAM_WEBHOOK_SECRET in .env.example.

    Error policy: a payload problem is answered 200 (a retry cannot help, and a
    failing status makes Telegram retry the same update forever); anything that
    could still succeed on a retry — a failed spawn, a missing secret — is
    answered 5xx, so the update is kept rather than discarded. Dedup-Dict
    failures are answered 200 because by then the update has been accepted and
    the turn is already queued; there is nothing left to retry.
    """
    import hmac

    expected_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if not expected_secret:
        # Misconfiguration, not a bad update. Answer 5xx on purpose so Telegram
        # queues and retries rather than every message being dropped while the
        # secret is unset. Set it and re-run set_webhook.py; the queue drains.
        print("[telegram] TELEGRAM_WEBHOOK_SECRET is unset — refusing the update")
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"ok": False, "error": "misconfigured"})

    if not hmac.compare_digest(secret_header or "", expected_secret):
        # Not from our Telegram webhook (or a stale secret). Drop it silently
        # with a 200: a prober learns nothing, and Telegram does not retry a
        # genuinely-wrong secret forever. This is the check that stops forgeries.
        print("[telegram] secret-token mismatch — dropped")
        return {"ok": True}

    message = (payload or {}).get("message") or {}
    text = message.get("text")
    if not text:
        # Ignore non-text updates (edits, button taps, other update types).
        return {"ok": True}

    allowed_user_id = os.environ.get("TELEGRAM_ALLOWED_USER_ID")
    if not allowed_user_id:
        # Misconfiguration, not a bad update. Answer 5xx on purpose: a 200 here
        # would tell Telegram the update is delivered and every message would
        # be discarded silently. This way they queue until the secret is fixed.
        print("[telegram] TELEGRAM_ALLOWED_USER_ID is unset — refusing the update")
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"ok": False, "error": "misconfigured"})

    # isinstance guard: a forged (or malformed) payload can send `from` as a
    # non-object; `{}.get` on an int raised and returned 500 before this.
    sender = message.get("from")
    sender_id = str(sender.get("id", "")) if isinstance(sender, dict) else ""
    if sender_id != allowed_user_id:
        # Silently drop — defence in depth behind the secret check above.
        return {"ok": True}

    update_id = (payload or {}).get("update_id")
    update_id = update_id if isinstance(update_id, int) else None
    if update_id is not None and _telegram_seen(deps.claims_kv, update_id, "webhook"):
        print(f"[telegram] duplicate update_id {update_id} — acked, not spawned")
        return {"ok": True, "duplicate": True}

    # Spawn FIRST, record second. If .spawn raises, nothing has been recorded,
    # so Telegram's retry of this same update runs instead of being dismissed
    # as a duplicate — which would lose the message for good, since Telegram
    # treats our 5xx as "try again" and would have already been told the update
    # was seen. The reverse order has no such recovery.
    spawn(text, update_id)
    if update_id is not None:
        _telegram_record(deps.claims_kv, update_id, "webhook")
    return {"ok": True}
