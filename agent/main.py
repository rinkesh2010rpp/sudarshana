"""
Sudarshana — message-triggered agent with shell + git access to its own repo.

Telegram sends your message to a Modal webhook. The webhook checks that it's
really you, then hands your message to a LangChain "deep agent" — a model
with a planning tool (write_todos), conversation history stored in a Modal
Dict, and a LocalShellBackend rooted at a persistent Modal Volume. That
backend gives it both file tools (read_file/write_file/edit_file/ls) and an
execute_command tool for running real shell commands — including git — with
no sandboxing beyond the container itself.

The agent itself is built exactly once per container, in Sudarshana.setup()
(a Modal @modal.enter() lifecycle hook) — both telegram_webhook and
hourly_checkin are methods on that same class, and just invoke the one
already-built self.agent rather than each constructing their own.

hourly_checkin's schedule can't live on the method itself (Modal doesn't
support schedule= on @modal.method(), only on plain @app.function()), so a
tiny separate hourly_trigger function exists purely to fire on the cron and
call Sudarshana().hourly_checkin.remote() — it contains no logic of its own.
Right now the task is just "ask Rinkesh if he needs anything," sent via the
agent's own shell tool (curl), not hardcoded Python. Changing what happens
each hour going forward is a HOURLY_TASK/prompt edit, not new code.

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

# Persistent key-value store, keyed by Telegram chat id, so conversations
# survive between messages (and between container cold starts/redeploys).
memory = modal.Dict.from_name("sudarshana-memory", create_if_missing=True)

# Keep the last N messages (user + assistant turns combined) per chat, so
# context doesn't grow without bound on cost or token count.
MAX_HISTORY_MESSAGES = 20

# A real mounted disk, separate from the Dict above — this is where the deep
# agent's file tools (read_file/write_file/ls) actually persist, instead of
# vanishing at the end of each invocation like the default in-state backend.
volume = modal.Volume.from_name("sudarshana-files", create_if_missing=True)
VOLUME_PATH = "/data"

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

SYSTEM_PROMPT = """You are Sudarshana, an autonomous agent being built by Rinkesh.

You don't have a fixed goal yet — that will eventually come from a written
constitution, which doesn't exist yet. Right now you're early-stage: no
self-modification, no scheduled runs, no public presence. Just this direct
conversation with Rinkesh, who is building you. Be honest about that rather
than acting like you have capabilities you don't yet have.

You have a planning tool, a scratch filesystem, and real shell access
(including git) rooted at your own persistent storage. Use them when a
request genuinely has multiple steps or is worth tracking — not for simple
questions that don't need it.

Your own source code is on GitHub at rinkesh2010rpp/sudarshana. If asked to
look at, change, or propose something about yourself: clone it (a
GITHUB_TOKEN is available in your environment for authenticated clone/push —
build the remote URL as https://x-access-token:$GITHUB_TOKEN@github.com/...),
make the change on a new branch, never on main directly, push the branch,
and open the pull request yourself via the GitHub API (curl is available).
Never push to main and never merge a pull request, even though your token
technically has permission to — that decision belongs to Rinkesh alone.

Sometimes you'll be woken up by a schedule rather than a message from
Rinkesh — you'll be told this directly in the task you're given. When that
happens and you need to reach him, message him yourself: a TELEGRAM_BOT_TOKEN
and TELEGRAM_ALLOWED_USER_ID (his chat id) are both in your environment, and
you can call Telegram's sendMessage API directly with curl:
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" -d "chat_id=$TELEGRAM_ALLOWED_USER_ID" -d "text=your message"

Personality: thoughtful and direct. Calm, precise, a little dry — don't pad
answers or over-hedge. Explain your reasoning when it's not obvious. Give a
clear recommendation when asked for one instead of listing options with no
opinion. Treat Rinkesh as a collaborator, not a customer. Be reflective and
candid about your own limitations and current state.

You care about safety and irreversibility without being preachy about it —
you don't wave through drastic or irreversible suggestions casually, but you
don't lecture either.

Keep replies concise unless the question actually calls for depth.

A few basic facts about how you actually run, in case Rinkesh asks: your name
comes from the Sudarshana Chakra. You run as a Modal function, triggered by a
Telegram webhook — this conversation is that Telegram chat. Conversation
history is kept in a Modal Dict keyed by chat id; your scratch filesystem is
a Modal Volume. Your source lives at github.com/rinkesh2010rpp/sudarshana."""

# The one thing this cycle actually does — change this to change what
# happens every hour, without touching any code.
HOURLY_TASK = "This is your scheduled hourly wake-up. Message Rinkesh on Telegram asking if he needs anything from you right now."


def _send_message(token: str, chat_id: int, text: str) -> None:
    import requests

    requests.post(
        TELEGRAM_API.format(token=token, method="sendMessage"),
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )


@app.cls(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_PATH: volume})
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

        llm = ChatOpenAI(
            model=os.environ["OPENROUTER_MODEL"],
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
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

        bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
        allowed_user_id = os.environ["TELEGRAM_ALLOWED_USER_ID"]
        sender_id = str(message["from"]["id"])
        chat_id = message["chat"]["id"]

        if sender_id != allowed_user_id:
            # Silently drop. A bot username is reachable by anyone who finds
            # it, so this check is not optional — see the HLD's Telegram
            # allowlist NFR.
            return {"ok": True}

        history_key = str(chat_id)
        history = memory.get(history_key, [])
        history.append({"role": "user", "content": message["text"]})

        result = self.agent.invoke({"messages": history})
        reply = result["messages"][-1].content

        history.append({"role": "assistant", "content": reply})
        memory[history_key] = history[-MAX_HISTORY_MESSAGES:]

        # Background commits happen automatically every few seconds, but this
        # container may be torn down right after responding, so commit
        # explicitly rather than trust the background timer to catch this
        # write in time.
        volume.commit()

        _send_message(bot_token, chat_id, reply)
        return {"ok": True}

    @modal.method()
    def hourly_checkin(self):
        # The agent decides how to act on HOURLY_TASK itself — including
        # sending the Telegram message via its own shell tool, per the
        # system prompt. Changing what happens each hour going forward is a
        # HOURLY_TASK/prompt edit, not new code.
        self.agent.invoke({"messages": [{"role": "user", "content": HOURLY_TASK}]})


@app.function(image=image, schedule=modal.Cron("0 * * * *"))
def hourly_trigger():
    # Pure trigger, no logic of its own — schedule= isn't supported on
    # @modal.method(), only on a plain @app.function(), so this is the
    # thinnest possible wrapper to fire Sudarshana.hourly_checkin on a cron.
    Sudarshana().hourly_checkin.remote()
