"""Public, read-only turn status (/api/status, /api/events) as a plain FastAPI app."""
from sudarshana.memory.status import STATUS_HISTORY_LIMIT, _status_ts
def create_status_app(deps):
    """status-center tier 1: read-only /api/status + /api/events.

    A separate web endpoint on the same Modal app, reading the shared
    sudarshana-status Dict live (mounts NO volume). Public + read-only like the
    gateway blog: every payload is own-work summaries only, no secrets or
    infra details. CORS is enabled via explicit middleware so the gateway SPA
    can fetch cross-origin.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI(title="sudarshana-status")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def _snapshot():
        d = deps.status_kv
        current = d.get("current", {}) or {}
        history = d.get("history", []) or []
        last = history[0] if history else None
        # Public API: strip internal keys like _turn_key.
        current = {k: v for k, v in current.items() if not k.startswith("_")}
        return {
            "generated_at": _status_ts(),
            "current": current,
            "last_turn": last,
            "history_count": len(history),
        }

    @web_app.get("/api/status")
    def api_status():
        return _snapshot()

    @web_app.get("/api/events")
    def api_events(limit: int = 50):
        d = deps.status_kv
        history = d.get("history", []) or []
        limit = max(1, min(int(limit), STATUS_HISTORY_LIMIT))
        return {
            "generated_at": _status_ts(),
            "events": history[:limit],
            "current_trace": d.get("current_trace", []) or [],
        }

    return web_app
