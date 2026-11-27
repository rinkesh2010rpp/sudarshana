"""Per-turn timing logs and the full LLM-call log."""
import os
import time

from sudarshana.core.config import VOLUME_PATH

def _build_timing_handler():
    """Log every model call and tool call with its duration to modal app
    logs — the only visibility into which step of invoke() is slow."""
    import time

    from langchain_core.callbacks import BaseCallbackHandler

    class _TimingHandler(BaseCallbackHandler):
        def __init__(self):
            self.starts: dict = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            self.starts[run_id] = time.monotonic()
            print(f"[timing] model call started ({len(messages[0])} messages in context)")

        def on_llm_end(self, response, *, run_id, **kwargs):
            elapsed = time.monotonic() - self.starts.pop(run_id, time.monotonic())
            print(f"[timing] model call finished in {elapsed:.1f}s")

        def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
            self.starts[run_id] = time.monotonic()
            print(f"[timing] tool '{serialized.get('name', '?')}' started: {input_str[:200]!r}")

        def on_tool_end(self, output, *, run_id, **kwargs):
            elapsed = time.monotonic() - self.starts.pop(run_id, time.monotonic())
            print(f"[timing] tool finished in {elapsed:.1f}s")

    return _TimingHandler()


# Every model call (main agent, sub-agents, summarization) recorded in full —
# input messages, output message, token usage, latency — as a future training /
# eval dataset. One gzipped JSONL file per turn, so overlapping turns in two
# containers never write the same file (the Volume is last-commit-wins):
#   /data/llm-calls/<YYYY-MM-DD>/<HHMMSS>-<trigger>-<id>.jsonl.gz
# Tool schemas are identical call-to-call and large, so each distinct set is
# written once to /data/llm-calls/tools/<hash>.json and records carry the hash.
# Committed by the turn's existing volume.commit().
LLM_CALLS_DIR = os.path.join(VOLUME_PATH, "llm-calls")


def _build_call_logger(trigger: str):
    import gzip
    import hashlib
    import json
    import uuid
    from datetime import datetime, timezone

    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import message_to_dict, messages_to_dict

    now = datetime.now(timezone.utc)
    turn_id = uuid.uuid4().hex[:12]
    day_dir = os.path.join(LLM_CALLS_DIR, now.strftime("%Y-%m-%d"))
    path = os.path.join(day_dir, f"{now.strftime('%H%M%S')}-{trigger}-{turn_id}.jsonl.gz")
    tools_dir = os.path.join(LLM_CALLS_DIR, "tools")

    def _tools_hash(tools) -> str | None:
        if not tools:
            return None
        blob = json.dumps(tools, sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
        tools_path = os.path.join(tools_dir, f"{digest}.json")
        if not os.path.exists(tools_path):
            os.makedirs(tools_dir, exist_ok=True)
            with open(tools_path, "w", encoding="utf-8") as f:
                f.write(blob)
        return digest

    class _CallLogger(BaseCallbackHandler):
        def __init__(self):
            self.pending: dict = {}
            self.seq = 0

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                                invocation_params=None, metadata=None, **kwargs):
            try:
                params = dict(invocation_params or {})
                tools = params.pop("tools", None)
                # Keep generation settings only — nothing credential-shaped.
                for k in ("api_key", "openai_api_key", "default_headers", "http_client"):
                    params.pop(k, None)
                self.pending[run_id] = {
                    "started": time.time(),
                    "record": {
                        "turn_id": turn_id,
                        "trigger": trigger,
                        "run_id": str(run_id),
                        "parent_run_id": str(parent_run_id) if parent_run_id else None,
                        "node": (metadata or {}).get("langgraph_node"),
                        "params": params,
                        "tools_hash": _tools_hash(tools),
                        "input": messages_to_dict(messages[0]),
                    },
                }
            except Exception as e:  # never let logging break a turn
                print(f"[llm-calls] start capture failed: {e!r}")

        def _finish(self, run_id, **fields):
            entry = self.pending.pop(run_id, None)
            if entry is None:
                return
            try:
                rec = entry["record"]
                self.seq += 1
                rec["seq"] = self.seq
                rec["ts"] = datetime.now(timezone.utc).isoformat()
                rec["latency_s"] = round(time.time() - entry["started"], 3)
                rec.update(fields)
                os.makedirs(day_dir, exist_ok=True)
                # Append mode writes one gzip member per record; gzip readers
                # read concatenated members as one stream, and a turn that
                # dies mid-run keeps every call logged before it.
                with gzip.open(path, "at", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            except Exception as e:
                print(f"[llm-calls] write failed: {e!r}")

        def on_llm_end(self, response, *, run_id, **kwargs):
            try:
                gen = response.generations[0][0]
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) or (response.llm_output or {}).get("token_usage")
                self._finish(
                    run_id,
                    output=message_to_dict(msg) if msg is not None else {"text": gen.text},
                    usage=dict(usage) if usage else None,
                )
            except Exception as e:
                print(f"[llm-calls] end capture failed: {e!r}")
                self.pending.pop(run_id, None)

        def on_llm_error(self, error, *, run_id, **kwargs):
            self._finish(run_id, error=repr(error))

    return _CallLogger()
