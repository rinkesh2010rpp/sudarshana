"""Remembering which incoming messages were already handled.

One rolling list of recent message ids per scope, in one KVStore entry each.
A channel picks its own scope (Telegram uses "seen:webhook" and "seen:run"),
so channels and gates never consume each other's claims. These raise on a
store error; callers decide whether that fails open.
"""


def is_seen(kv, scope: str, msg_id) -> bool:
    return msg_id in list(kv.get(scope) or [])


def mark_seen(kv, scope: str, msg_id, window: int) -> None:
    seen = [i for i in (kv.get(scope) or []) if i != msg_id]
    seen.append(msg_id)
    kv.put(scope, seen[-window:])
