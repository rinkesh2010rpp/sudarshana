"""
Sudarshana — step 1: message-triggered agent, no cron, no Netlify.

Telegram sends your message to a Modal webhook. The webhook checks that it's
really you, then hands your message to a LangChain "deep agent" — a model
with a planning tool (write_todos), a real persistent filesystem backed by a
Modal Volume, and subagent delegation, on top of a personality system prompt
and conversation history stored in a Modal Dict. Still no repo access, no
self-modification — those come later.

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

image = modal.Image.debian_slim().pip_install(
    "requests",
    "fastapi[standard]",
    "deepagents",
    "langchain-openai",
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

You have a planning tool and a scratch filesystem available. Use them when a
request genuinely has multiple steps or is worth tracking — not for simple
questions that don't need it.

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


def _send_message(token: str, chat_id: int, text: str) -> None:
    import requests

    requests.post(
        TELEGRAM_API.format(token=token, method="sendMessage"),
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )


def _build_agent(api_key: str, model: str):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return create_deep_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=VOLUME_PATH),
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_PATH: volume},
)
@modal.fastapi_endpoint(method="POST")
def telegram_webhook(payload: dict):
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
        # Silently drop. A bot username is reachable by anyone who finds it,
        # so this check is not optional — see the HLD's Telegram allowlist NFR.
        return {"ok": True}

    history_key = str(chat_id)
    history = memory.get(history_key, [])
    history.append({"role": "user", "content": message["text"]})

    agent = _build_agent(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ["OPENROUTER_MODEL"],
    )
    result = agent.invoke({"messages": history})
    reply = result["messages"][-1].content

    history.append({"role": "assistant", "content": reply})
    memory[history_key] = history[-MAX_HISTORY_MESSAGES:]

    # Background commits happen automatically every few seconds, but this
    # container may be torn down right after responding, so commit explicitly
    # rather than trust the background timer to catch this write in time.
    volume.commit()

    _send_message(bot_token, chat_id, reply)
    return {"ok": True}
