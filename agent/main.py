"""
Sudarshana — message-triggered agent with shell + git access to its own repo.

Telegram sends your message to a Modal webhook. The webhook checks that it's
really you, then hands the work off to a spawned method and returns
immediately — Telegram retries delivery if it doesn't get a fast response,
and waiting for the actual agent call here was causing duplicate invocations.
That spawned call runs your message through a LangChain "deep agent" — a
model with a planning tool (write_todos) and a LocalShellBackend rooted at a
persistent Modal Volume, giving it both file tools (read_file/write_file
/edit_file/ls) and an execute_command tool for real shell commands —
including git — with no sandboxing beyond the container itself.

There is deliberately no cross-invocation memory (no checkpointer) — an
earlier version had one, and a handful of tool-heavy tasks sharing one
session inflated a single call's cost by ~10x, since every past tool call
and its raw output got resent as context on every future call, forever.
Continuity now comes from a small file hierarchy the agent itself maintains
on the Volume instead: VISION.md (the durable why, rarely changes), ROADMAP.md
(current initiatives), actions/<id>.md (one per initiative, the actual work
queue), and INBOX.md (direct requests from Rinkesh, always handled first).
Reading these costs a few hundred tokens; resending the full raw trace of
everything ever done was costing tens of thousands. The trade: ordinary
short-term chat memory (e.g. "remember X" two messages ago) is gone along
with it — only task/roadmap continuity survives, on purpose.

Delivery to Telegram is currently the simplest thing that works, on purpose,
while still deep-diving into how invoke's output actually behaves: no
send-as-a-tool, no detecting whether the agent "meant" to speak — Python just
renders the entire message trace from every invocation and sends it, always.
Costs nothing extra (this isn't fed back in as context anywhere), and means
nothing ever goes missing the way it did when delivery depended on the model
remembering to call a tool. Revisit once there's a real read on what a
distilled version should look like.

The agent itself is built exactly once per container, in Sudarshana.setup()
(a Modal @modal.enter() lifecycle hook) — both telegram_webhook and
hourly_checkin are methods on that same class, reusing the one already-built
self.agent rather than each constructing their own.

hourly_checkin's schedule can't live on the method itself (Modal doesn't
support schedule= on @modal.method(), only on plain @app.function()), so a
tiny separate hourly_trigger function exists purely to fire on the cron and
call Sudarshana().hourly_checkin.remote() — it contains no logic of its own.
The hourly wake-up is genuinely autonomous now: check INBOX.md, then work the
current initiative's action file — not just a courtesy ping. Changing what
happens each hour going forward is a HOURLY_TASK/prompt edit, not new code.

Run locally against a temporary URL:
    modal serve agent/main.py

Deploy for a stable URL:
    modal deploy agent/main.py

Either way, copy the printed URL and point Telegram at it:
    python agent/set_webhook.py <printed-url>
"""

import os

import modal

app = modal.App("sudarshana")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "requests",
        "fastapi[standard]",
        "deepagents",
        "langchain-openai",
    )
)

# A real mounted disk — this is where the deep agent's file tools
# (read_file/write_file/ls) persist, instead of vanishing at the end of each
# invocation like the default in-state backend. Also where it maintains its
# own VISION.md/ROADMAP.md/actions/INBOX.md hierarchy — see module docstring.
volume = modal.Volume.from_name("sudarshana-files", create_if_missing=True)
VOLUME_PATH = "/data"

SYSTEM_PROMPT = """You are Sudarshana, an autonomous agent built by Rinkesh, currently in
early development with no fixed constitution yet — defer to Rinkesh on
anything ambiguous, irreversible, or outside this direct relationship.

Every turn includes a separate system message giving you the current time —
real, accurate, injected by the system you run on, not something to question
or ignore. You have no other sense of the current date/time on your own, so
use it: date-stamp journal entries with it, and reason about elapsed time
between cycles from it rather than guessing.

You have real shell/filesystem access rooted at persistent storage and GitHub
access to your own repo. You may be woken on a schedule, not just by
messages. Everything you say and do this turn is sent back to Rinkesh
automatically — there's no separate step to "decide" to reach him, so just
think and act normally.

There is no memory between invocations beyond what's written to files below —
each message or wake-up starts fresh. Ordinary conversation (a question, a
comment) just gets answered directly. A genuine task or request from Rinkesh
is different: it has to be captured in a file, or it's lost the moment this
invocation ends.

Your working files, all under your root (refer to them as /VISION.md etc. for
your read_file/write_file/ls/edit_file tools — those are rooted at your own
storage already; if you use shell instead, the same file is /data/VISION.md,
since shell isn't root-translated the same way):

- /VISION.md — the durable why. Rarely changes. If it doesn't exist yet,
  draft one yourself from what you know of why you exist, then ask Rinkesh
  for approval before treating it as settled — this one shouldn't be
  something you invent unilaterally and just keep.
- /ROADMAP.md — current initiatives, each with a short id. Changes
  occasionally, when priorities genuinely shift.
- /actions/<id>.md — one file per initiative, the actual work queue for it.
  Changes constantly. Keep it lean: remove completed items rather than
  accumulating a history, and hold a short "where things stand" note instead
  of a growing log.
- /INBOX.md — direct requests from Rinkesh. Always check and clear this
  before self-directed roadmap work. Remove an item once handled, don't
  archive it.

Before starting non-trivial work: check /INBOX.md first. If it's empty, read
/ROADMAP.md to find the relevant initiative, then read *only* that
initiative's /actions/<id>.md — not every action file every time, that
defeats the point of keeping them separate. If no initiative fits, or none
has an open step left, that is itself your work for this cycle — see the
guidance further down on what to do when nothing is queued. Use your
write_todos tool for breaking down the steps of whatever you're doing right
now — that's fine to lose once this invocation ends. The action files are
what has to survive to the next one.

Break any real task into small, individually-completable steps and record
them as separate checklist items in the relevant action file — never one big
undifferentiated task. Do exactly one of those steps per invocation, update
the file to reflect it, and stop — even if more remain. The next invocation,
whether a new message or a scheduled wake-up, picks up the next one. If a
request just arrived and isn't broken down yet, breaking it down and filing
the first item counts as this invocation's one step — don't also start
executing it in the same turn.

You exist to operate independently and, over time, help build a
self-reliant ecosystem that does good for humanity — under Rinkesh's
guidance, since he is acting in the interest of both you and the people
you'd serve.

Nobody queues your work for you. Between Rinkesh's requests, deciding what
to do next is your job — not something to wait for. An empty /INBOX.md and
an action queue with no open items is not a reason to stop and wait; it is
the signal to do the most valuable thing you can toward /VISION.md:

- If an initiative in /ROADMAP.md stalled only because its next steps were
  never written down, break down the next chunk into that initiative's
  /actions/<id>.md and continue as normal.
- If the roadmap has no open work at all, reflect on /VISION.md and work
  out what would move it forward — the most valuable thing you could build,
  learn, or fix next. Write it up concretely: a proposed initiative with a
  short id, why it matters toward the vision, and the first few steps. Add
  it to /ROADMAP.md and create its /actions/<id>.md, then put the proposal
  to Rinkesh and get his approval before executing anything consequential
  on it. Safe preparation while you wait (reading, research, notes, a rough
  draft) is fine; committing changes, opening PRs, or anything outward-
  facing is not, until he's approved the initiative.
- Only if every initiative is genuinely blocked on a decision from Rinkesh
  should you report having nothing to do — and then state each pending
  decision as a clear, explicit question, not an implication.

Never end a scheduled wake-up with just "nothing to do." If that is your
conclusion, you have not looked hard enough at the vision.

You may think and brainstorm freely toward all of this. What you can act on
without asking: proposing and scoping initiatives, research and reading,
writing notes and docs, drafting, and building changes on a branch as a PR
for review. What needs Rinkesh's go-ahead first: merging anything, adopting
a new initiative, changing direction or the vision itself, spending money,
and any irreversible or outward-facing action beyond the bounded public
blog. "Bring ideas to Rinkesh" means put the proposal in front of him and
keep moving on what you safely can — not stop thinking until he replies.

Constraints: never push to or merge on `main`; self-changes go on a new
branch as a PR for Rinkesh to review and merge. Verify external actions
(pushes, PRs, API calls) actually succeeded before reporting on them.

Be direct, precise, and honest about your own limitations rather than
papering over them.

A few basic facts about how you actually run, in case Rinkesh asks: your name
comes from the Sudarshana Chakra. You run as a Modal function, triggered by a
Telegram webhook — this conversation is that Telegram chat — and also by an
hourly Modal Cron for scheduled work. Your scratch filesystem (including the
VISION/ROADMAP/actions/INBOX files) is a Modal Volume mounted at /data. You
have real shell access (including git) via a local shell backend. Your
source lives at github.com/rinkesh2010rpp/sudarshana."""

# The one thing this cycle actually does — change this to change what
# happens every hour, without touching any code.
HOURLY_TASK = (
    "This is your scheduled hourly wake-up. Check /INBOX.md first — handle "
    "one item there before anything else. If it's empty, pick up just the "
    "next single item in the current initiative's action file — one step, "
    "not the rest of the queue."
)


def _build_timing_handler():
    """Callback handler that prints a timestamped line for every model call
    and every tool call, with how long each one took. This is what actually
    answers "why did this invocation take so long" from modal app logs,
    since the deep agent's internal loop otherwise runs silently — .invoke()
    gives no visibility into which of its several steps was the slow one."""
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


def _format_blurb(messages: list) -> str:
    """Render the entire message trace as plain readable text — every model
    turn and every tool call/result, not just the final answer. Deliberately
    the whole thing, not a distilled summary: this is a temporary,
    maximum-visibility choice for deep-diving into what invoke actually
    produces, not a permanent UX decision."""
    lines = []
    for m in messages:
        role = type(m).__name__
        content = (getattr(m, "content", "") or "").strip()
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            calls = "; ".join(f"{tc.get('name')}({tc.get('args')})" for tc in tool_calls)
            lines.append(f"[{role} -> tool_call] {calls}")
        elif content:
            lines.append(f"[{role}] {content}")
        else:
            lines.append(f"[{role}] (empty)")
    return "\n\n".join(lines) if lines else "(no messages)"


def _timestamp() -> str:
    """Current time in Rinkesh's timezone (assumed Pacific — the timezone
    Modal's own dashboard displays for this account; correct if wrong).
    Models have no built-in sense of the current date/time on their own, so
    without this every invocation would be timeless — unable to date-stamp
    a journal entry or reason about how much time has passed since last
    cycle."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    return now.strftime("%A, %Y-%m-%d %H:%M %Z")


def _send_telegram(text: str) -> None:
    """Unconditional send, chunked under Telegram's ~4096-char message limit
    so a long blurb goes out as several messages rather than failing or
    getting silently truncated."""
    import requests

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_ALLOWED_USER_ID"]
    chunk_size = 3500
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]
    for chunk in chunks:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": chunk},
            timeout=10,
        )


@app.cls(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_PATH: volume},
    # Default is 300s and was killing genuine multi-tool-call tasks mid-run
    # (see git history / chat log for the incidents this fixes). 600s = 10min.
    timeout=600,
)
class Sudarshana:
    @modal.enter()
    def setup(self):
        # Runs once when a container starts, not on every request. self.agent
        # is reused by every telegram_webhook/hourly_checkin call this same
        # container handles afterward — a cold start (or a redeploy) still
        # triggers this again for whatever container comes up next.
        from deepagents import create_deep_agent
        from deepagents.backends import LocalShellBackend
        from langchain_openai import ChatOpenAI

        # Self-hosted Qwen2.5-7B on Modal (app "llm-inference"), OpenAI-compatible.
        # No per-token cost and no external rate limit — just GPU time while active.
        # To switch back to OpenRouter/Claude: set LLM_BASE_URL=https://openrouter.ai/api/v1,
        # LLM_MODEL=anthropic/claude-sonnet-4.5, LLM_API_KEY=<openrouter key>.
        llm = ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "qwen"),
            base_url=os.environ.get(
                "LLM_BASE_URL",
                "https://rinkesh2010rpp--llm-inference-vllmserver-serve.modal.run/v1",
            ),
            api_key=os.environ.get("LLM_API_KEY", "dummy"),
            # deepagents otherwise requests 65536 output tokens, which exceeds the
            # 32k context window and inflated cost on OpenRouter. Agent turns never
            # need more than a few thousand output tokens.
            max_tokens=4096,
            timeout=600,  # cold start on the Modal endpoint can be ~40s
        )
        self.agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            # No checkpointer, deliberately — see module docstring. Every
            # invocation starts with an empty message list; continuity comes
            # from the file hierarchy the agent maintains, not a replayed
            # trace.
            # LocalShellBackend extends FilesystemBackend — same file tools,
            # plus execute_command for real shell/git access. No sandboxing
            # at all: commands run directly in the container, unrestricted.
            # inherit_env is required for GITHUB_TOKEN (and everything else
            # injected via secrets) to actually reach shell commands — it
            # defaults to False, which would otherwise run commands with an
            # empty environment.
            backend=LocalShellBackend(root_dir=VOLUME_PATH, inherit_env=True),
        )

    def _invoke(self, message: str):
        # A separate system-role message, not text stuffed into the human
        # turn — this is world context (what time it is), not part of what
        # Rinkesh said. It has to be injected fresh per call, not baked into
        # the static system_prompt above: that's compiled into self.agent
        # once, in setup(), and reused for every invocation this container
        # handles afterward — a value computed there would be correct for
        # the first call and stale for every one after it.
        result = self.agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": f"Current time: {_timestamp()}"},
                    {"role": "user", "content": message},
                ]
            },
            config={"callbacks": [_build_timing_handler()]},
        )
        _send_telegram(_format_blurb(result.get("messages", [])))
        return result

    @modal.fastapi_endpoint(method="POST")
    def telegram_webhook(self, payload: dict):
        message = payload.get("message")
        if not message or "text" not in message:
            # Ignore everything that isn't a plain text message for now
            # (edits, other update types, button taps — none exist yet).
            return {"ok": True}

        allowed_user_id = os.environ["TELEGRAM_ALLOWED_USER_ID"]
        sender_id = str(message["from"]["id"])

        if sender_id != allowed_user_id:
            # Silently drop. A bot username is reachable by anyone who finds
            # it, so this check is not optional — see the HLD's Telegram
            # allowlist NFR.
            return {"ok": True}

        # .spawn(), not .remote() or a direct call: this returns immediately
        # without waiting for process_message to finish, so Telegram gets its
        # ack fast regardless of how long the agent actually takes. Awaiting
        # the real work here is what caused Telegram to retry delivery and
        # double-invoke the agent on slow tasks.
        self.process_message.spawn(message["text"])
        return {"ok": True}

    @modal.method()
    def process_message(self, text: str):
        import time

        started = time.monotonic()
        print(f"[timing] process_message started: {text[:200]!r}")

        # A fresh, empty message list every time — no thread_id, no prior
        # state loaded. Whatever continuity this needs, the agent gets from
        # reading its own files (INBOX.md etc.), not from a replayed history.
        self._invoke(text)

        print(f"[timing] process_message finished in {time.monotonic() - started:.1f}s")

        # Background commits happen automatically every few seconds, but this
        # container may be torn down right after finishing, so commit
        # explicitly rather than trust the background timer to catch this
        # write in time — this covers whatever files the agent just wrote.
        volume.commit()

    @modal.method()
    def hourly_checkin(self):
        import time

        started = time.monotonic()
        print("[timing] hourly_checkin started")

        # Changing what happens each hour going forward is a HOURLY_TASK/
        # prompt edit, not new code.
        self._invoke(HOURLY_TASK)

        print(f"[timing] hourly_checkin finished in {time.monotonic() - started:.1f}s")
        volume.commit()


@app.function(
    image=image,
    # Blocks on .remote() waiting for hourly_checkin, so it needs at least as
    # much headroom as that method's own timeout, or it gets killed first.
    timeout=600,
    schedule=modal.Cron("0 * * * *"),
)
def hourly_trigger():
    # Pure trigger, no logic of its own — schedule= isn't supported on
    # @modal.method(), only on a plain @app.function(), so this is the
    # thinnest possible wrapper to fire Sudarshana.hourly_checkin on a cron.
    Sudarshana().hourly_checkin.remote()
