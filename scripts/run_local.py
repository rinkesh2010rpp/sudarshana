"""Run the agent on this machine, without Modal: one turn per message.

    SUDARSHANA_DATA=./local-data python scripts/run_local.py "your message"
    SUDARSHANA_DATA=./local-data python scripts/run_local.py        # prompt loop

Needs the packages in requirements.txt and the model settings in the
environment (USE_OPENROUTER=1, OPENROUTER_MODEL, OPENROUTER_API_KEY).

WARNING: the agent gets an unsandboxed shell on THIS machine, rooted at
SUDARSHANA_DATA but free to leave it, with this process's environment
(minus SHELL_WITHHELD_SECRETS). Run it somewhere you're fine with that.

Local differences: the status, dedup and a2a stores live in memory and are
gone when it exits; owner messages (asks, the reply) print to the console;
the prompt still names /data paths in its text.
"""
import os
import pathlib
import sys

if not os.environ.get("SUDARSHANA_DATA"):
    sys.exit("Set SUDARSHANA_DATA to a folder for the agent's workspace (it must not be /data).")
root = pathlib.Path(os.environ["SUDARSHANA_DATA"]).resolve()
root.mkdir(parents=True, exist_ok=True)
os.environ["SUDARSHANA_DATA"] = str(root)  # read once, when sudarshana.core.config is imported
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sudarshana.core.agent import build_agent, run_turn  # noqa: E402
from sudarshana.core.deps import Deps  # noqa: E402
from sudarshana.infra.storage.local.stores import DictKV, LocalStorage  # noqa: E402
from sudarshana.infra.storage.sqlite.checkpoints import SqliteCheckpoints  # noqa: E402
from sudarshana.interfaces.console import ConsoleNotifier  # noqa: E402


def main() -> None:
    deps = Deps(
        storage=LocalStorage(str(root)),
        checkpoints=SqliteCheckpoints(str(root / "checkpoints.db")),
        status_kv=DictKV(),
        claims_kv=DictKV(),
        a2a_kv=DictKV(),
        owner_notifier=ConsoleNotifier(),
    )
    agent = build_agent(deps)
    messages = [" ".join(sys.argv[1:])] if sys.argv[1:] else None
    while True:
        text = messages.pop(0) if messages else input("you> ").strip()
        if not text:
            continue
        result = run_turn(agent, text, "local")
        print(f"\nsudarshana> {result.reply}\n")
        if messages is not None and not messages:
            break


if __name__ == "__main__":
    main()
