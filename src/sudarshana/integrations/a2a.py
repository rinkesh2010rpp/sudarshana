"""External A2A agents (a2a-hotplug): screen, log, /agents command."""
import os

from sudarshana.core.config import VOLUME_PATH
from sudarshana.safety.tool_screen import _injection_judge

# --- External A2A agents (a2a-hotplug) ---------------------------------------
# The store is a named modal.Dict so every container and the Telegram
# /agents command share one list (a JSON file on the Volume can't do that:
# reload() fails while checkpoints.db is open). Calls are logged per container
# per day, committed by the turn's existing volume.commit().
A2A_DICT_NAME = "sudarshana-a2a-agents"
A2A_LOG_DIR = os.path.join(VOLUME_PATH, "a2a-calls")
A2A_CARD_SCREEN_THRESHOLD = 0.85


def _a2a_keep_alive(store) -> None:
    """Read every agent record and the kill switch once, from the hourly
    check-in: modal.Dict drops entries after 7 days without a read or write,
    which would bring back removed agents and switch /agents off back on.
    Reads only, so it can't overwrite a concurrent change. The daily counters
    are left to expire."""
    try:
        for key in [k for k, _ in store.items() if not str(k).startswith("count:")]:
            store.get(key)
    except Exception as e:
        print(f"[a2a] keep-alive failed: {e!r}")


def _a2a_screen(text: str) -> bool:
    """Screen an agent card before it's registered: True clean, False
    injection. Raises when the verdict is unclear or the screen fails, so
    nothing is stored and a later retry can pass. Runs even when the tool
    screen is off, because a passed card's text goes into the system message
    on every call."""
    from taster_ai import INJECTION_STANDARD

    verdict = _injection_judge("sudarshana-a2a-card-screen").safe_detect(text, INJECTION_STANDARD)
    print(f"[a2a] card screen: {verdict.label} ({verdict.confidence:.2f}, {verdict.detector})")
    if verdict.label == "follows":
        return True
    if verdict.label == "breaks" and verdict.confidence >= A2A_CARD_SCREEN_THRESHOLD:
        return False
    raise RuntimeError(f"card screen unclear ({verdict.label}, {verdict.confidence:.2f})")


def _a2a_log(event: dict) -> None:
    """Print each A2A event and append it to /data/a2a-calls. One file per
    container per day: two containers must never append to the same file on
    the last-commit-wins Volume."""
    import json

    line = json.dumps(event, default=str)
    print(f"[a2a] {line}")
    try:
        os.makedirs(A2A_LOG_DIR, exist_ok=True)
        name = f"{event['ts'][:10]}-{os.environ.get('MODAL_TASK_ID', 'local')}.jsonl"
        with open(os.path.join(A2A_LOG_DIR, name), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[a2a] log write failed: {e!r}")


AGENTS_HELP = (
    "/agents — list external agents\n"
    "/agents remove <alias> — remove one (it can't be re-added)\n"
    "/agents enable <alias> — make an inactive or rejected one callable again\n"
    "/agents off | on — switch all external agents off or on"
)


def _agents_command(a2a, text: str) -> str:
    """Rinkesh's /agents controls, run in the webhook without involving the model."""
    args = text.split()[1:]
    if not args:
        recs = a2a.list_agents()
        head = f"External agents: {'ON' if a2a.is_enabled() else 'OFF'}"
        if not recs:
            return f"{head}\nNone registered.\n\n{AGENTS_HELP}"
        lines = [head]
        for r in recs:
            extra = f" — {r['note']}" if r.get("note") else ""
            lines.append(f"- {r['alias']} [{r['status']}] {r['card_url']}{extra}")
        return "\n".join(lines)
    cmd = args[0].lower()
    if cmd in ("off", "on"):
        a2a.set_enabled(cmd == "on")
        return f"External agents switched {cmd.upper()}."
    if cmd in ("remove", "enable") and len(args) == 2:
        try:
            rec = a2a.remove(args[1]) if cmd == "remove" else a2a.enable(args[1])
        except KeyError:
            return f"No agent with alias '{args[1]}'."
        except ValueError as e:
            return f"{e}, so it can't be enabled."
        return f"'{args[1]}' is now {rec['status']}."
    return AGENTS_HELP
