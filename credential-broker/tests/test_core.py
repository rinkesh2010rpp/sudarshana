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


def test_post_to_a_registered_host_attaches_the_key(active):
    site = FakeSite((201, {"post": {"id": "p1"}}))
    url = "https://www.moltbook.com/api/v1/posts"
    out = req(active, site, url=url, method="POST", body={"title": "hi"})
    assert out["status"] == "ok" and out["http_status"] == 201
    assert site.calls[0]["method"] == "POST"
    assert site.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"
    assert json.loads(site.calls[0]["body"]) == {"title": "hi"}


def test_unsupported_method_refused(active):
    site = FakeSite()
    assert req(active, site, url="https://www.moltbook.com/x", method="TRACE")["status"] == "refused"
    assert site.calls == []


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


# --- recovery without a human ------------------------------------------------

UUID = "3f2a9c4e-1b7d-4e8a-9f6b-2c5d8e1a7b30"
NEW_KEY = "moltbook_sk_NEW9876543210zyxwvuZYXWVU"


class AuthSite:
    """Signup returns a key; other calls succeed only with the valid key."""

    def __init__(self, signup_key=KEY, valid_key=KEY):
        self.signup_key, self.valid_key = signup_key, valid_key
        self.signups = 0

    def __call__(self, method, host, port, path, headers, body):
        if path.endswith("/register"):
            self.signups += 1
            return 201, {}, json.dumps({"agent": {"api_key": self.signup_key, "id": UUID}}).encode()
        ok = headers.get("Authorization") == f"Bearer {self.valid_key}"
        return (200 if ok else 401), {}, b'{"r": 1}'


def test_wrong_field_recovers_by_itself(vault):
    site = AuthSite()
    assert reg(vault, site, {**RECIPE, "secret": "agent.id"})["status"] == "ok"
    out = req(vault, site, url="https://www.moltbook.com/api/v1/x")
    assert out["http_status"] == 401 and "rejected the key" in out["note"]
    assert vault.hosts["www.moltbook.com"]["key_rejected"] is True

    out = reg(vault, site, RECIPE)  # corrected path, no replace
    assert out["status"] == "ok" and site.signups == 1  # site not contacted again
    no_key_anywhere(out)
    assert req(vault, site, url="https://www.moltbook.com/api/v1/x")["http_status"] == 200
    assert "key_rejected" not in vault.hosts["www.moltbook.com"]


def test_a_working_key_cannot_be_replaced(vault):
    site = AuthSite()
    reg(vault, site)
    assert req(vault, site, url="https://www.moltbook.com/api/v1/x")["http_status"] == 200
    assert reg(vault, site, {**RECIPE, "secret": "agent.id"})["status"] == "refused"
    assert handle({"op": "register", "recipe": RECIPE, "replace": True}, vault,
                  fetch=site, log=lambda *_: None)["status"] == "refused"
    assert site.signups == 1 and vault.hosts["www.moltbook.com"]["key"] == KEY


def test_swap_to_a_non_key_keeps_the_old_entry(vault):
    site = AuthSite()
    reg(vault, site, {**RECIPE, "secret": "agent.id"})
    req(vault, site, url="https://www.moltbook.com/api/v1/x")  # 401
    out = reg(vault, site, {**RECIPE, "secret": "agent.nope"})
    assert out["status"] == "error" and "agent.api_key" in out["fields"]
    assert vault.hosts["www.moltbook.com"]["key"] == UUID  # unchanged, still rejected


def test_revoked_key_is_replaced_with_a_fresh_signup(vault):
    site = AuthSite()
    reg(vault, site)
    site.valid_key = None  # the site revokes the key
    assert req(vault, site, url="https://www.moltbook.com/api/v1/x")["http_status"] == 401
    site.signup_key = site.valid_key = NEW_KEY
    out = handle({"op": "register", "recipe": RECIPE, "replace": True}, vault,
                 fetch=site, log=lambda *_: None)
    assert out["status"] == "ok" and site.signups == 2
    assert NEW_KEY not in json.dumps(out)
    assert vault.hosts["www.moltbook.com"]["key"] == NEW_KEY
    assert req(vault, site, url="https://www.moltbook.com/api/v1/x")["http_status"] == 200


def test_needs_human_host_can_be_retried_with_replace(vault):
    reg(vault, FakeSite(TimeoutError("read timed out")))
    out = reg(vault, FakeSite())
    assert out["status"] == "needs_human" and "replace: true" in out["hint"]
    out = handle({"op": "register", "recipe": RECIPE, "replace": True}, vault,
                 fetch=FakeSite((201, REGISTERED)), log=lambda *_: None)
    assert out["status"] == "ok" and vault.hosts["www.moltbook.com"]["state"] == "active"


def test_replace_must_be_a_real_boolean(vault):
    # A string "true" is not consent to sign up again, even on a rejected key.
    site = AuthSite()
    reg(vault, site)
    site.valid_key = None
    req(vault, site, url="https://www.moltbook.com/api/v1/x")  # 401: key rejected
    out = handle({"op": "register", "recipe": RECIPE, "replace": "true"}, vault,
                 fetch=site, log=lambda *_: None)
    assert out["status"] == "error" and site.signups == 1  # treated as a swap, no signup


# --- pinned_fetch: what counts as "not sent" (no network) --------------------


class FakeConn:
    """Stands in for urllib3's HTTPSConnection."""
    fail_connect = None
    fail_response = None
    requests = []
    timeouts = []

    def __init__(self, host, **kw):
        self.sock = None
        self.timeout = kw.get("timeout")

    def connect(self):
        if FakeConn.fail_connect:
            raise FakeConn.fail_connect

    def request(self, method, path, body=None, headers=None, preload_content=True):
        FakeConn.requests.append((method, path))
        FakeConn.timeouts.append(self.timeout)

    def getresponse(self):
        if FakeConn.fail_response:
            raise FakeConn.fail_response
        class R:
            status = 200
            headers = {"Content-Type": "application/json"}
            def read(self, amt):
                return b'{"ok": true}'
        return R()

    def close(self):
        pass


@pytest.fixture
def fake_conn(monkeypatch):
    import urllib3.connection
    FakeConn.fail_connect = FakeConn.fail_response = None
    FakeConn.requests = []
    FakeConn.timeouts = []
    monkeypatch.setattr(urllib3.connection, "HTTPSConnection", FakeConn)
    return FakeConn


PUBLIC = fake_resolver("93.184.215.14")


def test_handshake_failure_is_not_sent(fake_conn):
    import urllib3
    fake_conn.fail_connect = urllib3.exceptions.SSLError("certificate verify failed")
    with pytest.raises(NotSent):
        core.pinned_fetch("POST", "www.moltbook.com", 443, "/r", {}, b"{}", resolver=PUBLIC)
    assert fake_conn.requests == []


def test_failure_after_sending_is_uncertain(fake_conn):
    fake_conn.fail_response = TimeoutError("read timed out")
    with pytest.raises(TimeoutError):
        core.pinned_fetch("POST", "www.moltbook.com", 443, "/r", {}, b"{}", resolver=PUBLIC)
    assert fake_conn.requests == [("POST", "/r")]


def test_read_timeout_is_20s_and_paths_are_encoded(fake_conn):
    core.pinned_fetch("GET", "www.moltbook.com", 443, "/s?q=a b&u=café&x=%41", {}, None,
                      resolver=PUBLIC)
    assert fake_conn.requests[-1] == ("GET", "/s?q=a%20b&u=caf%C3%A9&x=%41")
    assert fake_conn.timeouts[-1] == core.READ_TIMEOUT_S


URL = "https://www.moltbook.com/api/v1/x"


def test_403_does_not_mark_the_key_rejected(active):
    req(active, FakeSite((403, {"e": "not your post"})), url=URL, method="DELETE")
    assert "key_rejected" not in active.hosts["www.moltbook.com"]
    assert reg(active, FakeSite())["status"] == "refused"


@pytest.mark.parametrize("failure", [
    (409, {"e": "name taken"}),
    (429, {"e": "slow down"}),
    (500, {"e": "boom"}),
    NotSent("host did not resolve"),
    TimeoutError("result lost"),
])
def test_failed_replace_keeps_the_old_key(vault, failure):
    site = AuthSite()
    reg(vault, site)
    site.valid_key = None
    req(vault, site, url=URL)  # 401: key rejected
    out = handle({"op": "register", "recipe": RECIPE, "replace": True}, vault,
                 fetch=FakeSite(failure), log=lambda *_: None)
    assert out["status"] == "error"
    entry = vault.hosts["www.moltbook.com"]
    assert entry["state"] == "active" and entry["key"] == KEY
    assert entry.get("key_rejected") is True  # can still replace again


def test_header_only_fix_swaps_without_signing_up(vault):
    site = AuthSite()
    reg(vault, site, {**RECIPE, "attach": {"header": "X-API-Key", "scheme": ""}})
    assert req(vault, site, url=URL)["http_status"] == 401
    assert reg(vault, site, RECIPE)["status"] == "ok" and site.signups == 1
    assert req(vault, site, url=URL)["http_status"] == 200


def test_signup_with_failed_handshake_leaves_the_host_free(vault, fake_conn):
    import urllib3
    fake_conn.fail_connect = urllib3.exceptions.SSLError("certificate verify failed")
    fetch = lambda *a: core.pinned_fetch(*a, resolver=PUBLIC)  # noqa: E731
    assert reg(vault, fetch)["status"] == "error"
    assert "www.moltbook.com" not in vault.hosts
    fake_conn.fail_connect = None
    assert reg(vault, FakeSite((201, REGISTERED)))["status"] == "ok"
