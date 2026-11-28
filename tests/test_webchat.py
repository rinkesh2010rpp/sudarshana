"""The web chat channel (interfaces/webchat/chat.py) on the test server.

Drives the real routes and chat_turn with fake deps and run_turn stubbed.
No model, no network, no Modal.

Why: the chat page replaces Telegram for testing on one machine. These pin
that a message is shown and queued (not run on the request), the reply and
anything sent to the owner land on the page, the "working" flag clears even
when a turn fails, and the turn is recorded in the status store like any other.
"""
import pytest
from fastapi.testclient import TestClient

from conftest import fake_deps
from sudarshana.core.agent import TurnResult
from sudarshana.infra.runtime.server import app as server
from sudarshana.infra.storage.local.stores import DictKV
from sudarshana.interfaces.webchat import chat as webchat


PASSWORD = "correct horse battery"


def _client(app, auth=None):
    c = TestClient(app)
    c.auth = auth
    return c


class _Agent:
    a2a = object()


@pytest.fixture()
def env():
    chat = webchat.WebChat()
    deps = fake_deps(lambda name: DictKV(), owner_notifier=webchat.WebChatNotifier(chat))
    queued = []
    client = _client(server.create_app(deps, queued.append, chat, PASSWORD), ("me", PASSWORD))
    return chat, deps, queued, client


def test_a_message_is_shown_and_queued_not_run(env):
    chat, deps, queued, client = env
    assert client.post("/chat/send", json={"text": "hello"}).json() == {"ok": True}
    assert queued == [("chat", "hello")]
    body = client.get("/chat/messages").json()
    assert [m["text"] for m in body["messages"]] == ["hello"] and body["working"] is True


def test_an_empty_message_is_refused(env):
    chat, deps, queued, client = env
    assert client.post("/chat/send", json={"text": "  "}).json()["ok"] is False
    assert queued == []


def test_the_chat_needs_the_password(env):
    chat, deps, queued, client = env
    anon = TestClient(client.app)
    for method, path in [("get", "/chat"), ("get", "/chat/messages"), ("post", "/chat/send")]:
        assert getattr(anon, method)(path).status_code == 401
    wrong = _client(client.app, ("me", "guess"))
    assert wrong.post("/chat/send", json={"text": "hi"}).status_code == 401
    assert queued == []
    # the status API is not behind it
    assert anon.get("/api/status").status_code == 200


def test_no_password_means_no_chat():
    deps = fake_deps(lambda name: DictKV())
    with pytest.raises(ValueError):
        server.create_app(deps, lambda job: None, webchat.WebChat(), "")


def test_the_page_and_the_status_api_are_both_served(env):
    chat, deps, queued, client = env
    assert "Sudarshana" in client.get("/chat").text
    assert client.get("/api/status").status_code == 200


def test_a_turn_puts_its_reply_on_the_page_and_is_recorded(env, monkeypatch):
    chat, deps, queued, client = env
    seen = {}

    def fake_run_turn(agent, message, trigger, turn_key=None):
        seen["key"] = turn_key
        return TurnResult(reply=f"reply to {message}", outcome="done")

    monkeypatch.setattr(webchat, "run_turn", fake_run_turn)
    client.post("/chat/send", json={"text": "hello"})
    webchat.chat_turn(_Agent(), deps, chat, "hello")
    body = client.get("/chat/messages?after=1").json()
    assert [m["text"] for m in body["messages"]] == ["reply to hello"] and body["working"] is False
    rec = deps.status_kv.get(seen["key"])
    assert rec["state"] == "running-chat" and rec["outcome"] == "done"
    assert deps.storage.commits == 1


def test_a_failed_turn_still_clears_working(env, monkeypatch):
    chat, deps, queued, client = env

    def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(webchat, "run_turn", boom)
    client.post("/chat/send", json={"text": "hello"})
    with pytest.raises(RuntimeError):
        webchat.chat_turn(_Agent(), deps, chat, "hello")
    body = client.get("/chat/messages").json()
    assert body["working"] is False
    assert "turn failed: RuntimeError: model down" in body["messages"][-1]["text"]


def test_owner_messages_land_on_the_page(env):
    chat, deps, queued, client = env
    deps.owner_notifier.send("an ask for you")
    assert [m["from"] for m in chat.after(0)] == ["sudarshana"]
    assert deps.owner_notifier.channel == "web chat"


def test_history_is_capped_and_ids_keep_counting(monkeypatch):
    monkeypatch.setattr(webchat, "MAX_MESSAGES", 3)
    chat = webchat.WebChat()
    for i in range(5):
        chat.add("you", f"m{i}")
    kept = chat.after(0)
    assert [m["text"] for m in kept] == ["m2", "m3", "m4"]
    assert [m["id"] for m in kept] == [3, 4, 5]
    assert chat.after(4) == [kept[-1]]


def test_an_oversized_message_is_refused(env, monkeypatch):
    chat, deps, queued, client = env
    monkeypatch.setattr(webchat, "MAX_MESSAGE_CHARS", 10)
    assert client.post("/chat/send", json={"text": "x" * 11}).json()["ok"] is False
    assert queued == [] and chat.after(0) == []


def test_working_follows_queued_and_finished_turns():
    chat = webchat.WebChat()
    assert not chat.working
    chat.turn_queued()
    chat.turn_queued()
    chat.turn_done()
    assert chat.working
    chat.turn_done()
    chat.turn_done()  # never goes below zero
    assert not chat.working


def test_the_agent_shell_never_gets_the_chat_password():
    from sudarshana.safety import tool_screen

    assert "WEBCHAT_PASSWORD" in tool_screen.SHELL_WITHHELD_SECRETS
