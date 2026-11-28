"""Web chat: the owner's chat as a browser page, an alternative to Telegram.

The page lives at /chat on the server. You type; the turn runs on the worker;
the reply appears. Anything Sudarshana sends the owner unprompted (asks, the
hourly report) appears there too, through WebChatNotifier.

Every /chat route asks for a password (HTTP Basic auth: the browser's own
prompt; any username, only the password is checked). Still serve it on
127.0.0.1 and reach it through an SSH tunnel: anyone who gets past the
password can drive an agent with a shell.

The page's history is kept in memory, the last MAX_MESSAGES of it (a restart
clears the page); the conversation itself is in the checkpoint store, as with
Telegram.
"""
import secrets
import threading
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse

from sudarshana.core.agent import run_turn
from sudarshana.integrations.a2a import _agents_command
from sudarshana.memory.status import _status_turn_end, _status_turn_start

# Messages the page keeps; older ones drop off (the conversation itself is in
# the checkpoint store). Ids keep counting up, so the page never re-shows one.
MAX_MESSAGES = 500
# A longer message is refused rather than put into the model's context whole.
MAX_MESSAGE_CHARS = 20_000


class WebChat:
    """The page's message list and how many sent messages are still being
    worked on. Shared by the HTTP routes and the worker thread, so every
    change goes through the lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._messages = []
        self._next_id = 1
        self._pending = 0

    def add(self, sender: str, text: str) -> dict:
        with self._lock:
            msg = {"id": self._next_id, "from": sender, "text": text,
                   "ts": time.strftime("%H:%M:%S")}
            self._next_id += 1
            self._messages = (self._messages + [msg])[-MAX_MESSAGES:]
            return msg

    def after(self, last_id: int) -> list:
        with self._lock:
            return [m for m in self._messages if m["id"] > last_id]

    def turn_queued(self) -> None:
        with self._lock:
            self._pending += 1

    def turn_done(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)

    @property
    def working(self) -> bool:
        with self._lock:
            return self._pending > 0


class WebChatNotifier:
    """Notifier port: a message to the owner shows up on the chat page."""

    channel = "web chat"

    def __init__(self, chat: WebChat):
        self.chat = chat

    def send(self, text: str) -> None:
        self.chat.add("sudarshana", text)


def chat_turn(agent, deps, chat: WebChat, text: str) -> None:
    """One turn for a chat message: the steps around run_turn.

    No dedup gate: the page sends each message once, it never resends."""
    try:
        if text.strip().lower().split()[:1] == ["/agents"]:
            chat.add("sudarshana", _agents_command(agent.a2a, text))
            return
        started = time.monotonic()
        print(f"[timing] chat turn started: {text[:200]!r}")
        turn_key = _status_turn_start(deps, "running-chat")
        result = run_turn(agent, text, "chat", turn_key)
        # The message came in on the chat page, so the reply goes back there.
        chat.add("sudarshana", result.reply)
        print(f"[timing] chat turn finished in {time.monotonic() - started:.1f}s")
        _status_turn_end(deps, turn_key, result.outcome, "chat turn finished")
        deps.storage.commit()
    except Exception as e:
        # Show the failure on the page rather than going quiet. The full
        # traceback is in the server log (the worker prints it).
        chat.add("sudarshana", f"[turn failed: {type(e).__name__}: {str(e)[:300]} — see the server log]")
        raise
    finally:
        chat.turn_done()


def chat_routes(chat: WebChat, enqueue, password: str) -> APIRouter:
    """GET /chat (the page), POST /chat/send, GET /chat/messages?after=N, all
    behind `password`. `enqueue(("chat", text))` hands the turn to the
    runtime's worker."""
    basic = HTTPBasic()

    def require_password(creds: HTTPBasicCredentials = Depends(basic)) -> None:
        if not secrets.compare_digest(creds.password.encode(), password.encode()):
            raise HTTPException(status_code=401, detail="wrong password",
                                headers={"WWW-Authenticate": "Basic"})

    web = APIRouter(dependencies=[Depends(require_password)])

    @web.get("/chat", response_class=HTMLResponse)
    def page():
        return PAGE

    @web.post("/chat/send")
    def send(body: dict):
        text = str((body or {}).get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "empty message"}
        if len(text) > MAX_MESSAGE_CHARS:
            return {"ok": False, "error": f"message over {MAX_MESSAGE_CHARS} characters"}
        chat.add("you", text)
        chat.turn_queued()
        enqueue(("chat", text))
        return {"ok": True}

    @web.get("/chat/messages")
    def messages(after: int = 0):
        return {"messages": chat.after(after), "working": chat.working}

    return web


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sudarshana chat</title>
<style>
  :root { --bg:#fafafa; --fg:#1a1a1a; --muted:#666; --you:#dbeafe; --her:#fff; --line:#ddd; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16181c; --fg:#e8e8e8; --muted:#999; --you:#1e3a5f; --her:#22252b; --line:#333; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--fg);
         display:flex; flex-direction:column; height:100vh; }
  header { padding:10px 16px; border-bottom:1px solid var(--line); font-weight:600; }
  header small { font-weight:400; color:var(--muted); }
  #log { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:80%; padding:8px 12px; border-radius:10px; border:1px solid var(--line);
         white-space:pre-wrap; overflow-wrap:anywhere; }
  .you { align-self:flex-end; background:var(--you); }
  .sudarshana { align-self:flex-start; background:var(--her); }
  .ts { display:block; font-size:12px; color:var(--muted); margin-top:4px; }
  #working { padding:0 16px 6px; color:var(--muted); min-height:1.4em; }
  form { display:flex; gap:8px; padding:12px 16px; border-top:1px solid var(--line); }
  textarea { flex:1; resize:none; height:64px; padding:8px; font:inherit; color:inherit;
             background:var(--her); border:1px solid var(--line); border-radius:8px; }
  button { padding:0 18px; font:inherit; border-radius:8px; border:1px solid var(--line); cursor:pointer; }
</style>
</head>
<body>
<header>Sudarshana <small>private chat</small></header>
<div id="log"></div>
<div id="working"></div>
<form id="f">
  <textarea id="t" placeholder="Message Sudarshana (Enter sends, Shift+Enter for a new line)"></textarea>
  <button type="submit">Send</button>
</form>
<script>
let last = 0;
const log = document.getElementById("log"), t = document.getElementById("t");
function show(m) {
  const d = document.createElement("div");
  d.className = "msg " + (m.from === "you" ? "you" : "sudarshana");
  d.textContent = m.text;
  const s = document.createElement("span");
  s.className = "ts"; s.textContent = m.ts;
  d.appendChild(s); log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}
async function poll() {
  try {
    const r = await fetch("/chat/messages?after=" + last);
    const j = await r.json();
    for (const m of j.messages) { show(m); last = m.id; }
    document.getElementById("working").textContent = j.working ? "Sudarshana is working\\u2026" : "";
  } catch (e) {
    document.getElementById("working").textContent = "Server not reachable.";
  }
}
async function send(e) {
  e && e.preventDefault();
  const text = t.value.trim();
  if (!text) return;
  t.value = "";
  await fetch("/chat/send", {method: "POST", headers: {"Content-Type": "application/json"},
                             body: JSON.stringify({text})});
  poll();
}
document.getElementById("f").addEventListener("submit", send);
t.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) send(e); });
poll(); setInterval(poll, 2000);
</script>
</body>
</html>
"""
