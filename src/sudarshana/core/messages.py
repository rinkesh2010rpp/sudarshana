"""Reading the agent's messages: text, final reply, trace blurb, clock."""

def _content_text(m) -> str:
    """A message's text. Content is a list of blocks, not a string, once an
    image is involved (the agent viewing a screenshot); keep the text blocks
    and mark the rest."""
    content = getattr(m, "content", "") or ""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, dict):
            parts.append(f"[{block.get('type', 'block')}]")
    return "\n".join(p for p in parts if p).strip()


def _final_message(messages: list) -> str:
    """The agent's last spoken message — the final AIMessage with real
    content. Falls back to a marker so a silent turn still sends something
    to Telegram rather than nothing (the full trace is in the modal logs)."""
    for m in reversed(messages):
        if type(m).__name__ != "AIMessage":
            continue
        content = _content_text(m)
        if content:
            return content
    return "[cycle ended with no final message — see modal app logs for the trace]"


def _format_blurb(messages: list) -> str:
    """Render the full message trace (every model turn and tool call/result)
    as plain text for the modal logs — the full record behind the one-line
    Telegram message."""
    lines = []
    for m in messages:
        role = type(m).__name__
        content = _content_text(m)
        tool_calls = getattr(m, "tool_calls", None)
        # Providers put reasoning in different fields; check all four.
        _ak = getattr(m, "additional_kwargs", {}) or {}
        _rm = getattr(m, "response_metadata", {}) or {}
        reasoning = (
            _ak.get("reasoning") or _ak.get("reasoning_content")
            or _rm.get("reasoning") or _rm.get("reasoning_content")
        )
        if reasoning:
            lines.append(f"[{role} · thinking] {reasoning.strip()}")
        if tool_calls:
            calls = "; ".join(f"{tc.get('name')}({tc.get('args')})" for tc in tool_calls)
            lines.append(f"[{role} -> tool_call] {calls}")
        elif content:
            lines.append(f"[{role}] {content}")
        elif not reasoning:
            lines.append(f"[{role}] (empty)")
    return "\n\n".join(lines) if lines else "(no messages)"


def _timestamp() -> str:
    """Current time in Rinkesh's timezone (assumed Pacific, per Modal's
    dashboard). The model has no clock of its own, so without this every
    invocation is timeless."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    return now.strftime("%A, %Y-%m-%d %H:%M %Z")
