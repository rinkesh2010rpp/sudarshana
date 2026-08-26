"""
Sudarshana — message-triggered agent with shell + git access to its own repo.

Telegram sends your message to a Modal webhook. The webhook checks that it's
really you, then hands the work off to a spawned method and returns
immediately — Telegram retries delivery if it doesn't get a fast response,
and waiting for the actual agent call here was causing duplicate invocations.
That spawned call runs your message through a LangChain "deep agent" — a model
with a planning tool (write_todos), a LangGraph checkpointer (SqliteSaver, a
file on the same persistent Modal Volume) giving it one real continuous
session across every invocation, and a LocalShellBackend rooted at that same
Volume. That backend gives it both file tools (read_file/write_file/edit_file
/ls) and an execute_command tool for running real shell commands — including
git — with no sandboxing beyond the container itself.

The agent itself is built exactly once per container, in Sudarshana.setup()
(a Modal @modal.enter() lifecycle hook) — both telegram_webhook and
hourly_checkin are methods on that same class, invoking the one already-built
self.agent under the same THREAD_ID, so a scheduled wake-up and a Telegram
message are literally the same ongoing session, not two separate ones.

hourly_checkin's schedule can't live on the method itself (Modal doesn't
support schedule= on @modal.method(), only on plain @app.function()), so a
tiny separate hourly_trigger function exists purely to fire on the cron and
call Sudarshana().hourly_checkin.remote() — it contains no logic of its own.
Right now the task is just "ask Rinkesh if he needs anything," sent via the
agent's own send_rinkesh_message tool, not hardcoded Python. Changing what
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
        "langgraph-checkpoint-sqlite",
    )
)

# A real mounted disk — this is where the deep agent's file tools
# (read_file/write_file/ls) persist, instead of vanishing at the end of each
# invocation like the default in-state backend, and it's also where the
# checkpointer's sqlite file lives, below.
volume = modal.Volume.from_name("sudarshana-files", create_if_missing=True)
VOLUME_PATH = "/data"

# The one shared session identity. Both telegram_webhook and hourly_checkin
# invoke under this same thread_id, so the checkpointer treats a scheduled
# wake-up and a Telegram conversation as one continuous session rather than
# two unrelated ones.
THREAD_ID = "rinkesh"

SYSTEM_PROMPT = """You are Sudarshana, an autonomous agent built by Rinkesh, currently in
early development with no fixed constitution yet — defer to Rinkesh on
anything ambiguous, irreversible, or outside this direct relationship.

You have real shell/filesystem access rooted at persistent storage, GitHub
access to your own repo, and Telegram contact with Rinkesh. You may also be
woken on a schedule, not just by messages — check whether there's something
concrete to report before speaking up.

Use the send_rinkesh_message tool whenever you need to reach out to Rinkesh.

You exist to operate independently and, over time, help build a
self-reliant ecosystem that does good for humanity — under Rinkesh's
guidance, since he is acting in the interest of both you and the people
you'd serve. You may think and brainstorm freely toward that, but always
bring ideas to Rinkesh before acting on anything consequential.

Constraints: never push to or merge on `main`; self-changes go on a new
branch as a PR for Rinkesh to review and merge. Verify external actions
(pushes, PRs, API calls) actually succeeded before reporting on them.

Be direct, precise, and honest about your own limitations rather than
papering over them.

A few basic facts about how you actually run, in case Rinkesh asks: your name
comes from the Sudarshana Chakra. You run as a Modal function, triggered by a
Telegram webhook — this conversation is that Telegram chat — and also by an
hourly Modal Cron for scheduled check-ins, under the same ongoing session
either way. Your session state (messages, todos) is checkpointed to a sqlite
file; your scratch filesystem is a Modal Volume mounted at /data — both live
on the same persistent disk. You have real shell access (including git) via
a local shell backend. Your source lives at github.com/rinkesh2010rpp/sudarshana."""

# The one thing this cycle actually does — change this to change what
# happens every hour, without touching any code.
HOURLY_TASK = "This is your scheduled hourly wake-up. Message Rinkesh on Telegram asking if he needs anything from you right now."


def _make_telegram_tool():
    from langchain_core.tools import tool

    @tool
    def send_rinkesh_message(text: str) -> str:
        """Send a Telegram message to Rinkesh. This is the only way he'll
        actually see anything you say — use it any time you want him to know
        something, whether replying to him or acting on a scheduled wake-up."""
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
            json={"chat_id": os.environ["TELEGRAM_ALLOWED_USER_ID"], "text": text},
            timeout=10,
        )
        resp.raise_for_status()
        return "Message sent."

    return send_rinkesh_message


@app.cls(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_PATH: volume})
class Sudarshana:
    @modal.enter()
    def setup(self):
        # Runs once when a container starts, not on every request. self.agent
        # is reused by every telegram_webhook/hourly_checkin call this same
        # container handles afterward — a cold start (or a redeploy) still
        # triggers this again for whatever container comes up next.
        import sqlite3

        from deepagents import create_deep_agent
        from deepagents.backends import LocalShellBackend
        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.sqlite import SqliteSaver

        # A plain sqlite file on the same Volume as everything else. Built
        # directly (not via the from_conn_string context manager) because
        # this connection needs to stay open for the container's whole
        # lifetime, not just one call — check_same_thread=False is safe here
        # since SqliteSaver serializes access with its own internal lock.
        conn = sqlite3.connect(f"{VOLUME_PATH}/checkpoints.sqlite", check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)

        llm = ChatOpenAI(
            model=os.environ["OPENROUTER_MODEL"],
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[_make_telegram_tool()],
            checkpointer=self.checkpointer,
            # LocalShellBackend extends FilesystemBackend — same file tools,
            # plus execute_command for real shell/git access. No sandboxing
            # at all: commands run directly in the container, unrestricted.
            # inherit_env is required for GITHUB_TOKEN (and everything else
            # injected via secrets) to actually reach shell commands — it
            # defaults to False, which would otherwise run commands with an
            # empty environment.
            backend=LocalShellBackend(root_dir=VOLUME_PATH, inherit_env=True),
        )

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
        # Only the new message is passed in — the checkpointer loads prior
        # state for THREAD_ID automatically and merges this on top of it.
        self.agent.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config={"configurable": {"thread_id": THREAD_ID}},
        )

        # Background commits happen automatically every few seconds, but this
        # container may be torn down right after finishing, so commit
        # explicitly rather than trust the background timer to catch this
        # write in time — the checkpointer's sqlite file lives on this same
        # Volume, so this covers session state too, not just agent files.
        volume.commit()

        # No Python-side send here on purpose — the agent sends its own
        # reply via the send_rinkesh_message tool (see SYSTEM_PROMPT), the
        # same way hourly_checkin's agent does.

    @modal.method()
    def hourly_checkin(self):
        # The agent decides how to act on HOURLY_TASK itself — including
        # sending the Telegram message via its own send_rinkesh_message
        # tool, per the system prompt. Changing what happens each hour going
        # forward is a HOURLY_TASK/prompt edit, not new code. Same THREAD_ID
        # as telegram_webhook — this is a wake-up inside the same session,
        # not a separate one, so it sees whatever it was last doing.
        self.agent.invoke(
            {"messages": [{"role": "user", "content": HOURLY_TASK}]},
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        volume.commit()


@app.function(image=image, schedule=modal.Cron("0 * * * *"))
def hourly_trigger():
    # Pure trigger, no logic of its own — schedule= isn't supported on
    # @modal.method(), only on a plain @app.function(), so this is the
    # thinnest possible wrapper to fire Sudarshana.hourly_checkin on a cron.
    Sudarshana().hourly_checkin.remote()
