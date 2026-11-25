"""Broker core against a fake site: no network, no Modal."""

import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from broker import core  # noqa: E402
from broker.core import NotSent, Refused, Vault, handle  # noqa: E402

KEY = "moltbook_sk_0123456789abcdefABCDEF"
SIGNUP = "https://www.moltbook.com/api/v1/agents/register"
RECIPE = {
    "signup": {"url": SIGNUP, "method": "POST", "body": {"name": "sud", "description": "d"}},
    "secret": "agent.api_key",
    "attach": {"header": "Authorization", "scheme": "Bearer"},
}
REGISTERED = {
    "agent": {
        "api_key": KEY,
        "claim_url": "https://www.moltbook.com/claim/moltbook_claim_abcdefghijklmnop",
        "verification_code": "reef-X4B2",
    },
    "important": "SAVE YOUR API KEY!",
}


class FakeSite:
    """Records calls; answers with queued (status, body) pairs or raises."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, host, port, path, headers, body):
        self.calls.append({"method": method, "host": host, "path": path,
                           "headers": headers, "body": body})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        status, payload = r
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return status, {}, data


@pytest.fixture
def vault(tmp_path):
    return Vault(str(tmp_path / "vault.json"))


def reg(vault, site, recipe=RECIPE):
    return handle({"op": "register", "recipe": recipe}, vault, fetch=site, log=lambda *_: None)


def req(vault, site, **kw):
    return handle({"op": "request", **kw}, vault, fetch=site, log=lambda *_: None)


def no_key_anywhere(result):
    assert KEY not in json.dumps(result)


# --- register ----------------------------------------------------------------


def test_register_stores_key_and_returns_only_safe_fields(vault):
    site = FakeSite((201, REGISTERED))
    out = reg(vault, site)
    assert out["status"] == "ok"
    no_key_anywhere(out)
    assert out["body"]["agent"]["claim_url"].startswith("https://www.moltbook.com/claim/")
    assert out["body"]["agent"]["verification_code"] == "reef-X4B2"
    entry = vault.hosts["www.moltbook.com"]
    assert entry["state"] == "active" and entry["key"] == KEY
    assert entry["allowed_methods"] == ["GET"]
    assert json.loads(site.calls[0]["body"]) == {"name": "sud", "description": "d"}


def test_register_refuses_a_host_that_already_has_a_key(vault):
    reg(vault, FakeSite((201, REGISTERED)))
    site = FakeSite()
    out = reg(vault, site)
    assert out["status"] == "refused" and site.calls == []


def test_wrong_secret_path_returns_field_names_only_then_retry_uses_saved_response(vault):
    bad = {**RECIPE, "secret": "agent.verification_code"}  # too short to be a key
    site = FakeSite((201, REGISTERED))
    out = reg(vault, site, bad)
    assert out["status"] == "error" and out["tries_left"] == 2
    assert "agent.api_key" in out["fields"]
    no_key_anywhere(out)
    assert "reef-X4B2" not in json.dumps(out)  # names only, no values

    out = reg(vault, site, RECIPE)  # corrected path; site must not be called again
    assert out["status"] == "ok" and len(site.calls) == 1
    no_key_anywhere(out)


def test_url_named_as_secret_is_rejected(vault):
    out = reg(vault, FakeSite((201, REGISTERED)), {**RECIPE, "secret": "agent.claim_url"})
    assert out["status"] == "error"
    no_key_anywhere(out)


def test_three_bad_paths_needs_human(vault):
    site = FakeSite((201, REGISTERED))
    for path in ("nope", "agent.nope", "agent.verification_code"):
        out = reg(vault, site, {**RECIPE, "secret": path})
    assert out["status"] == "needs_human"
    assert reg(vault, site)["status"] == "needs_human" and len(site.calls) == 1


def test_mislabelled_field_cannot_leak_or_lose_the_real_key(vault):
    # The agent names a key-like field that isn't the key (a UUID id). The real
    # api_key must stay hidden from the agent AND survive in the vault.
    weird = {"agent": {"api_key": KEY, "id": "3f2a9c4e-1b7d-4e8a-9f6b-2c5d8e1a7b30"}}
    out = reg(vault, FakeSite((201, weird)), {**RECIPE, "secret": "agent.id"})
    assert out["status"] == "ok"
    no_key_anywhere(out)
    entry = vault.hosts["www.moltbook.com"]
    assert entry["saved_response"]["agent"]["api_key"] == KEY


def test_oversized_body_is_refused_without_locking_the_host(vault):
    big = {**RECIPE, "signup": {**RECIPE["signup"],
                                "body": {"x": "a" * (core.MAX_BODY_BYTES + 10)}}}
    site = FakeSite((201, REGISTERED))
    assert reg(vault, site, big)["status"] == "refused"
    assert "www.moltbook.com" not in vault.hosts and site.calls == []
    assert reg(vault, site)["status"] == "ok"


def test_signup_4xx_clears_and_allows_a_new_attempt(vault):
    out = reg(vault, FakeSite((409, {"error": "name taken"})))
    assert out["status"] == "error" and out["http_status"] == 409
    assert "www.moltbook.com" not in vault.hosts
    assert reg(vault, FakeSite((201, REGISTERED)))["status"] == "ok"


@pytest.mark.parametrize("response", [
    (500, {"error": "boom"}),
    (302, {}),
    (201, b"<html>not json</html>"),
    TimeoutError("read timed out"),
])
def test_uncertain_signups_need_a_human_and_never_retry(vault, response):
    site = FakeSite(response)
    assert reg(vault, site)["status"] == "needs_human"
    assert reg(vault, site)["status"] == "needs_human"
    assert len(site.calls) == 1


def test_failure_before_sending_clears_pending(vault):
    out = reg(vault, FakeSite(NotSent("host did not resolve")))
    assert out["status"] == "error"
    assert "www.moltbook.com" not in vault.hosts


def test_interrupted_signup_needs_human(vault):
    vault.hosts["www.moltbook.com"] = {"state": "pending", "recipe": RECIPE}
    site = FakeSite()
    assert reg(vault, site)["status"] == "needs_human" and site.calls == []


def test_pending_is_saved_before_the_signup_is_sent(tmp_path):
    path = str(tmp_path / "vault.json")
    vault = Vault(path)
    seen = {}

    def site(*_a):
        seen["state"] = json.load(open(path))["hosts"]["www.moltbook.com"]["state"]
        return 201, {}, json.dumps(REGISTERED).encode()

    reg(vault, site)
    assert seen["state"] == "pending"


@pytest.mark.parametrize("attach", [
    {"header": "Host", "scheme": "Bearer"},
    {"header": "Authorization\r\nX-Evil", "scheme": "Bearer"},
    {"header": "Authorization", "scheme": "Bearer\r\nX"},
])
def test_bad_attach_is_refused(vault, attach):
    site = FakeSite()
    assert reg(vault, site, {**RECIPE, "attach": attach})["status"] == "refused"
    assert site.calls == []


# --- request -----------------------------------------------------------------


@pytest.fixture
def active(vault):
    reg(vault, FakeSite((201, REGISTERED)))
    return vault


def test_request_attaches_key_to_its_host(active):
    site = FakeSite((200, {"posts": [{"id": "post_0123456789abcdefghijk", "title": "hi"}]}))
    out = req(active, site, url="https://www.moltbook.com/api/v1/posts?limit=5")
    assert out["status"] == "ok" and out["http_status"] == 200
    assert site.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"
    assert site.calls[0]["path"] == "/api/v1/posts?limit=5"
    # The shape rule is off for request: long IDs survive.
    assert out["body"]["posts"][0]["id"] == "post_0123456789abcdefghijk"


def test_request_removes_the_key_if_the_site_echoes_it(active):
    site = FakeSite((200, {"me": {"note": f"your key is {KEY}", "token": "abc"}}))
    out = req(active, site, url="https://www.moltbook.com/api/v1/agents/me")
    no_key_anywhere(out)
    assert out["body"]["me"]["token"] == "[removed]"


def test_post_refused_until_allowed_in_vault(active):
    site = FakeSite((200, {"ok": True}))
    url = "https://www.moltbook.com/api/v1/posts"
    assert req(active, site, url=url, method="POST", body={"t": 1})["status"] == "refused"
    active.hosts["www.moltbook.com"]["allowed_methods"] = ["GET", "POST"]
    assert req(active, site, url=url, method="POST", body={"t": 1})["status"] == "ok"


@pytest.mark.parametrize("url", [
    "https://moltbook.com/api/v1/posts",       # different host, no key
    "https://evil.example/steal",
])
def test_key_never_goes_to_another_host(active, url):
    site = FakeSite((200, {}))
    assert req(active, site, url=url)["status"] == "refused"
    assert site.calls == []


def test_request_refuses_pending_host(vault):
    vault.hosts["www.moltbook.com"] = {"state": "pending", "recipe": RECIPE}
    assert req(vault, FakeSite(), url="https://www.moltbook.com/x")["status"] == "refused"


def test_get_with_body_refused(active):
    assert req(active, FakeSite(), url="https://www.moltbook.com/x", body={"a": 1})["status"] == "refused"


def test_redirect_is_returned_not_followed(active):
    out = req(active, FakeSite((301, {})), url="https://www.moltbook.com/x")
    assert out["note"] == "redirect not followed"


# --- targets -----------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "http://www.moltbook.com/x",
    "https://user:pw@www.moltbook.com/x",
    "https://127.0.0.1/x",
    "https://www.moltbook.com:8443/x",
    "",
])
def test_bad_urls_refused(url):
    with pytest.raises(Refused):
        core.parse_target(url)


def fake_resolver(*addrs):
    def resolve(host, port, type=0):
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, type, 6, "", (a, port))
                for a in addrs]
    return resolve


@pytest.mark.parametrize("addr", ["127.0.0.1", "10.0.0.5", "169.254.169.254",
                                  "192.168.1.1", "100.64.0.1", "::1", "::ffff:127.0.0.1"])
def test_non_public_addresses_refused(addr):
    with pytest.raises(NotSent):
        core.resolve_public("x.example", 443, fake_resolver("93.184.215.14", addr))


def test_public_address_prefers_ipv4():
    assert core.resolve_public("x.example", 443,
                               fake_resolver("2606:2800:220:1::1", "93.184.215.14")) == "93.184.215.14"


def test_vault_survives_reload(tmp_path):
    path = str(tmp_path / "vault.json")
    v = Vault(path)
    reg(v, FakeSite((201, REGISTERED)))
    assert Vault(path).hosts["www.moltbook.com"]["key"] == KEY
