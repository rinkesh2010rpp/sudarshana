#!/usr/bin/env bash
# Run the server runtime in the background, in a tmux session, logging to a file.
#
#   bash src/sudarshana/infra/runtime/server/start.sh          start (no-op if running)
#   bash src/sudarshana/infra/runtime/server/start.sh stop     stop it
#   bash src/sudarshana/infra/runtime/server/start.sh status   is it running, and where
#   tmux attach -t sudarshana                                  watch it (Ctrl+B, D to leave)
#
# Settings come from the environment and from .env (read by the server itself):
#   PORT (8000), HOURLY (0), CHANNEL (webchat), SUDARSHANA_DATA (/data), PYTHON
#   (the repo's .venv if present, else python3). WEBCHAT_PASSWORD and the
#   model keys belong in .env. The log is $SUDARSHANA_DATA/server.log; set SUDARSHANA_DATA
#   in the environment, not only in .env, if the workspace isn't /data.
set -euo pipefail

SESSION=sudarshana
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/../../../.." && pwd)"   # .../src
REPO="$(dirname "$SRC")"
DATA="${SUDARSHANA_DATA:-/data}"
LOG="$DATA/server.log"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY="python3"
fi

command -v tmux >/dev/null || { echo "tmux is required (apt-get install tmux)" >&2; exit 1; }
running() { tmux has-session -t "$SESSION" 2>/dev/null; }

case "${1:-start}" in
  start)
    if running; then
      echo "already running (tmux attach -t $SESSION)"
      exit 0
    fi
    mkdir -p "$DATA"
    # Pass the settings on explicitly: a tmux server that was already running
    # would otherwise start the session with its own, older environment.
    # Never a secret here: this line is visible in `ps`. The password and keys
    # reach the server through .env, which it reads itself.
    envs=()
    for v in PORT HOURLY CHANNEL SUDARSHANA_DATA; do
      if [[ -n "${!v:-}" ]]; then envs+=("$v=$(printf '%q' "${!v}")"); fi
    done
    tmux new-session -d -s "$SESSION" \
      "cd $(printf '%q' "$SRC") && env ${envs[*]:-} $(printf '%q' "$PY") -u -m sudarshana.infra.runtime.server 2>&1 | tee -a $(printf '%q' "$LOG")"
    sleep 1
    if running; then
      echo "started: port ${PORT:-8000}, hourly ${HOURLY:-0}, channel ${CHANNEL:-webchat}"
      echo "log: $LOG   watch: tmux attach -t $SESSION"
    else
      echo "it exited at once; the reason is at the end of $LOG" >&2
      exit 1
    fi
    ;;
  stop)
    if running; then tmux kill-session -t "$SESSION" && echo "stopped"; else echo "not running"; fi
    ;;
  status)
    if running; then echo "running (tmux session $SESSION), log: $LOG"; else echo "not running"; fi
    ;;
  *)
    echo "usage: $0 [start|stop|status]" >&2
    exit 2
    ;;
esac
