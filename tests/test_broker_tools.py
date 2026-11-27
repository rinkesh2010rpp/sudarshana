"""The agent's broker tools: what they send to the credential broker, and the
guardrail settings around them. No network: requests.post is faked."""

import json

import pytest
import requests


from sudarshana.integrations import broker as broker_tools  # noqa: E402
from sudarshana.safety import tool_screen  # noqa: E402

TOKEN_ID, TOKEN_SECRET = "wk-test-id", "ws-test-secret"
URL = "https://broker.example.modal.run"


@pytest.fixture
def broker(monkeypatch):
    monkeypatch.setenv("BROKER_URL", URL)
    monkeypatch.setenv("BROKER_TOKEN_ID", TOKEN_ID)
    monkeypatch.setenv("BROKER_TOKEN_SECRET", TOKEN_SECRET)
    calls = []

    class Resp:
        status_code = 200

        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return Resp(calls_reply.pop(0) if calls_reply else {"status": "ok"})

    calls_reply = []
    monkeypatch.setattr(requests, "post", fake_post)
    return calls, calls_reply


RECIPE = {
    "signup": {"url": "https://www.moltbook.com/api/v1/agents/register",
               "method": "POST", "body": {"name": "sud"}},
    "secret": "agent.api_key",
    "attach": {"header": "Authorization", "scheme": "Bearer"},
}


def test_register_sends_the_recipe_with_the_proxy_token(broker):
    calls, reply = broker
    reply.append({"status": "ok", "body": {"agent": {"claim_url": "https://x/claim"}}})
    out = json.loads(broker_tools.broker_register(RECIPE))
    assert out["body"]["agent"]["claim_url"] == "https://x/claim"
    call = calls[0]
    assert call["url"] == URL
    assert call["json"] == {"op": "register", "recipe": RECIPE}
    assert call["headers"] == {"Modal-Key": TOKEN_ID, "Modal-Secret": TOKEN_SECRET}
    assert call["timeout"] == broker_tools.BROKER_TIMEOUT_S


def test_replace_is_sent_only_when_true(broker):
    calls, _ = broker
    broker_tools.broker_register(RECIPE, replace=True)
    broker_tools.broker_register(RECIPE, replace="true")
    assert calls[0]["json"]["replace"] is True
    assert "replace" not in calls[1]["json"]


def test_request_sends_url_method_and_optional_body(broker):
    calls, _ = broker
    broker_tools.broker_request("https://www.moltbook.com/api/v1/posts")
    broker_tools.broker_request("https://www.moltbook.com/api/v1/posts", "POST", {"title": "hi"})
    assert calls[0]["json"] == {"op": "request", "url": "https://www.moltbook.com/api/v1/posts",
                                "method": "GET"}
    assert calls[1]["json"]["method"] == "POST" and calls[1]["json"]["body"] == {"title": "hi"}


def test_missing_config_does_not_call_out(monkeypatch):
    for name in ("BROKER_URL", "BROKER_TOKEN_ID", "BROKER_TOKEN_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("called out"))
    out = json.loads(broker_tools.broker_request("https://www.moltbook.com/x"))
    assert out["status"] == "error" and "not configured" in out["reason"]


@pytest.mark.parametrize("failure", ["http_401", "network"])
def test_failures_never_echo_the_token(monkeypatch, failure):
    monkeypatch.setenv("BROKER_URL", URL)
    monkeypatch.setenv("BROKER_TOKEN_ID", TOKEN_ID)
    monkeypatch.setenv("BROKER_TOKEN_SECRET", TOKEN_SECRET)

    def fake_post(*a, **k):
        if failure == "network":
            raise requests.ConnectionError(f"failed with {TOKEN_SECRET}")

        class R:
            status_code = 401
        return R()

    monkeypatch.setattr(requests, "post", fake_post)
    text = broker_tools.broker_request("https://www.moltbook.com/x")
    assert json.loads(text)["status"] == "error"
    assert TOKEN_ID not in text and TOKEN_SECRET not in text


def test_token_is_withheld_from_the_shell():
    assert {"BROKER_TOKEN_ID", "BROKER_TOKEN_SECRET"} <= tool_screen.SHELL_WITHHELD_SECRETS


@pytest.mark.parametrize("scope,stage", [
    (tool_screen.CONSTITUTION_SCOPE, "tool_call"),
    (tool_screen.TOOL_SCREEN_DEFAULT_SCOPE, "tool_result"),
])
def test_both_tools_are_enforced_before_the_catch_all(scope, stage):
    targets = scope[stage]
    names = [t["tool"] for t in targets]
    for tool in ("broker_register", "broker_request"):
        t = targets[names.index(tool)]
        assert t["mode"] == "enforce" and t["on_error"] == "block"
        assert names.index(tool) < names.index("*")
