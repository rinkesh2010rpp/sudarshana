"""Credential broker core. No Modal imports; network only through `fetch`.

Two operations, both taking plain JSON:

    {"op": "register", "recipe": {...}}
    {"op": "request", "url": "...", "method": "GET", "body": {...}}

The broker signs up with the agent's recipe, keeps the key in the vault, and
attaches it only to that exact host. Nothing it returns contains a key.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from typing import Any, Callable
from urllib.parse import quote, urlsplit

MAX_RESPONSE_BYTES = 1_000_000
MAX_BODY_BYTES = 64_000
CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 20
MAX_PATH_TRIES = 3
REMOVED = "[removed]"
# Some sites (Moltbook among them) answer a bare library user agent with 503.
USER_AGENT = "Mozilla/5.0 (compatible; credential-broker/0.1)"

SECRET_NAMES = {"api_key", "key", "token", "access_token", "secret", "password"}
_SHAPE = re.compile(r"^[A-Za-z0-9_-]{20,}$")
_PATH = re.compile(r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*$")
_HEADER = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_SCHEME = re.compile(r"^[A-Za-z0-9_-]{0,32}$")
_FORBIDDEN_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "cookie"}
_SIGNUP_METHODS = {"POST", "PUT"}
_REQUEST_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


class Refused(Exception):
    """The broker won't do this. Nothing was sent."""


class NotSent(Exception):
    """Failed before any byte reached the site (DNS, address check)."""


# --- Vault -------------------------------------------------------------------


class Vault:
    """One JSON file: {"hosts": {host: entry}}. `commit` persists it (the Modal
    Volume commit in production, a no-op in tests)."""

    def __init__(self, path: str, commit: Callable[[], None] = lambda: None):
        self.path = path
        self._commit = commit
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {"hosts": {}}
        self.data.setdefault("hosts", {})

    @property
    def hosts(self) -> dict[str, dict]:
        return self.data["hosts"]

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
        self._commit()


# --- Target checks -----------------------------------------------------------


def parse_target(url: Any) -> tuple[str, int, str]:
    """https only, a hostname (not an IP), default port, no credentials in the
    URL. Returns (host, port, path_with_query)."""
    if not isinstance(url, str) or not url:
        raise Refused("url must be a non-empty string")
    try:
        p = urlsplit(url)
        port = p.port
    except ValueError:
        raise Refused("url is malformed")
    if p.scheme != "https":
        raise Refused("https only")
    if not p.hostname:
        raise Refused("url has no host")
    if p.username or p.password:
        raise Refused("credentials in the url are not allowed")
    if port not in (None, 443):
        raise Refused("only the default https port is allowed")
    host = p.hostname.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        raise Refused("use a hostname, not an IP address")
    except ValueError:
        pass
    path = p.path or "/"
    if p.query:
        path += "?" + p.query
    return host, 443, path


def _encode_path(path: str) -> str:
    """Percent-encode spaces and non-ASCII characters; leave existing %XX and
    URL punctuation alone (what urllib3's pool used to do for us)."""
    return quote(path, safe="!#$%&'()*+,/:;=?@[]~")


def resolve_public(host: str, port: int, resolver=socket.getaddrinfo) -> str:
    """Resolve and require every address to be public. Returns the one to
    connect to (IPv4 first). The caller connects to exactly this address."""
    try:
        infos = resolver(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise NotSent(f"host did not resolve ({type(e).__name__})")
    addrs = []
    for info in infos:
        raw = info[4][0].split("%")[0]
        ip = ipaddress.ip_address(raw)
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise NotSent("host resolves to a non-public address")
        addrs.append(ip)
    if not addrs:
        raise NotSent("host did not resolve")
    addrs.sort(key=lambda a: (a.version, str(a)))
    return str(addrs[0])


def pinned_fetch(method: str, host: str, port: int, path: str,
                 headers: dict[str, str], body: bytes | None,
                 resolver=socket.getaddrinfo) -> tuple[int, dict[str, str], bytes]:
    """One HTTPS request to the checked address, certificate verified against
    the hostname, no redirects, no retries, size-capped.

    The TCP connect and TLS handshake run as their own step. If either fails,
    not one byte of the request was written, so it raises NotSent and the
    caller may safely try again. Anything after that is uncertain."""
    import certifi
    from urllib3.connection import HTTPSConnection

    ip = resolve_public(host, port, resolver)
    conn = HTTPSConnection(
        ip,
        port=port,
        timeout=CONNECT_TIMEOUT_S,
        server_hostname=host,
        assert_hostname=host,
        cert_reqs="CERT_REQUIRED",
        ca_certs=certifi.where(),
    )
    try:
        try:
            conn.connect()
        except Exception as e:
            raise NotSent(f"could not connect ({type(e).__name__}); nothing was sent")
        # urllib3 re-applies conn.timeout to the socket on every call, so the
        # read timeout has to be set here, not on the socket.
        conn.timeout = READ_TIMEOUT_S
        conn.request(method, _encode_path(path), body=body,
                     headers={**headers, "Host": host}, preload_content=False)
        r = conn.getresponse()
        data = r.read(MAX_RESPONSE_BYTES + 1)
        status = r.status
        rheaders = {k.lower(): v for k, v in r.headers.items()}
    finally:
        conn.close()
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("response too large")
    return status, rheaders, data


# --- Redaction ---------------------------------------------------------------


def redact(obj: Any, exact: list[str], shape: bool) -> Any:
    """Replace secrets with [removed]: the exact stored key anywhere, fields
    named like secrets, and (register responses only) key-shaped strings."""
    exact = [e for e in exact if e]

    def scrub(s: str) -> str:
        for e in exact:
            s = s.replace(e, REMOVED)
        if shape and _SHAPE.match(s):
            return REMOVED
        return s

    def walk(v: Any) -> Any:
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                key = scrub(k) if isinstance(k, str) else k
                if isinstance(k, str) and k.lower() in SECRET_NAMES:
                    out[key] = REMOVED
                else:
                    out[key] = walk(val)
            return out
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str):
            return scrub(v)
        return v

    return walk(obj)


def field_paths(obj: Any, prefix: str = "") -> list[str]:
    """Every leaf's dotted path, no values."""
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out += field_paths(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    if isinstance(obj, list):
        out = []
        for i, v in enumerate(obj):
            out += field_paths(v, f"{prefix}.{i}" if prefix else str(i))
        return out
    return [prefix]


def extract(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def looks_like_key(v: Any) -> bool:
    return (
        isinstance(v, str)
        and len(v) >= 16
        and not re.search(r"\s", v)
        and not v.lower().startswith(("http://", "https://"))
    )


def _decode(data: bytes) -> Any:
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return data.decode("utf-8", errors="replace")


def _body_bytes(body: Any) -> bytes | None:
    if body is None:
        return None
    raw = json.dumps(body).encode()
    if len(raw) > MAX_BODY_BYTES:
        raise Refused("body too large")
    return raw


# --- Recipe ------------------------------------------------------------------


def validate_recipe(recipe: Any) -> dict:
    if not isinstance(recipe, dict):
        raise Refused("recipe must be an object")
    signup = recipe.get("signup")
    if not isinstance(signup, dict):
        raise Refused("recipe.signup must be an object")
    method = str(signup.get("method", "POST")).upper()
    if method not in _SIGNUP_METHODS:
        raise Refused("signup method must be POST or PUT")
    body = signup.get("body", {})
    if not isinstance(body, dict):
        raise Refused("signup body must be an object")
    secret = recipe.get("secret")
    if not isinstance(secret, str) or not _PATH.match(secret):
        raise Refused("recipe.secret must be a dotted path like agent.api_key")
    attach = recipe.get("attach")
    if not isinstance(attach, dict):
        raise Refused("recipe.attach must be an object")
    header = attach.get("header")
    scheme = attach.get("scheme", "")
    if not isinstance(header, str) or not _HEADER.match(header) or header.lower() in _FORBIDDEN_HEADERS:
        raise Refused("attach.header is not an allowed header name")
    if not isinstance(scheme, str) or not _SCHEME.match(scheme):
        raise Refused("attach.scheme must be a short word like Bearer, or empty")
    return {
        "signup": {"url": signup.get("url"), "method": method, "body": body},
        "secret": secret,
        "attach": {"header": header, "scheme": scheme},
    }


def _headers(extra: dict[str, str] | None = None, json_body: bool = False) -> dict[str, str]:
    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if json_body:
        h["Content-Type"] = "application/json"
    if extra:
        h.update(extra)
    return h


# --- Operations --------------------------------------------------------------


REPLACE_HINT = ("call register again with replace: true to sign up fresh; "
                "if the earlier signup did go through, this may create a second account")


def _needs_human(vault: Vault, host: str, reason: str) -> dict:
    entry = vault.hosts.setdefault(host, {})
    entry["state"] = "needs_human"
    entry["reason"] = reason
    vault.save()
    return {"status": "needs_human", "reason": reason, "hint": REPLACE_HINT}


def _abandon(vault: Vault, host: str, previous: dict | None) -> None:
    """Undo a pending signup: put back the entry it was replacing, if any."""
    if previous is not None:
        vault.hosts[host] = previous
    else:
        vault.hosts.pop(host, None)
    vault.save()


def _uncertain(vault: Vault, host: str, previous: dict | None, reason: str) -> dict:
    """The signup may have arrived. A first signup becomes needs_human; a
    replacement keeps the old key."""
    if previous is None:
        return _needs_human(vault, host, reason)
    _abandon(vault, host, previous)
    return {"status": "error", "reason": reason + "; your existing key was kept",
            "hint": REPLACE_HINT}


def _swap_key(host: str, entry: dict, recipe: dict, vault: Vault, log) -> dict:
    """The site rejected the stored key: take the key at the newly named path
    in the saved signup response. The site is not contacted."""
    saved = entry.get("saved_response")
    value = extract(saved, recipe["secret"]) if saved is not None else None
    if not looks_like_key(value):
        return {
            "status": "error",
            "reason": "no key at that path in the saved signup response; "
                      "try another path, or register with replace: true to sign up again",
            "fields": field_paths(saved) if saved is not None else [],
        }
    if value == entry.get("key") and recipe["attach"] == entry["recipe"]["attach"]:
        return {
            "status": "error",
            "reason": "this changes nothing (same key, same header); correct the "
                      "path or the header, or register with replace: true",
        }
    entry["key"] = value
    entry["recipe"] = recipe
    entry.pop("key_rejected", None)
    vault.save()
    log(f"[broker] register {host} swapped the key from the saved response")
    return {"status": "ok",
            "note": "key replaced from the saved signup response; the site was not contacted"}


def register(recipe: Any, vault: Vault, fetch, log, replace: bool = False) -> dict:
    recipe = validate_recipe(recipe)
    host, port, path = parse_target(recipe["signup"]["url"])
    entry = vault.hosts.get(host)

    if entry:
        state = entry.get("state")
        if state == "active":
            # A working key is never thrown away. Registering again is allowed
            # only after the site has rejected the stored key.
            if not entry.get("key_rejected"):
                raise Refused("this host already has a working key")
            if not replace:
                return _swap_key(host, entry, recipe, vault, log)
            # replace: fall through to a fresh signup
        elif state == "needs_human" and not replace:
            return {"status": "needs_human", "reason": entry.get("reason", ""), "hint": REPLACE_HINT}
        elif state == "pending" and "saved_response" not in entry and not replace:
            return _needs_human(vault, host, "signup was interrupted; the account may already exist")
        elif state == "pending" and not replace:
            # A retry with a corrected secret path: read the saved response,
            # never contact the site again.
            entry["recipe"] = recipe
            return _finish(host, entry, vault, log)

    # Every refusal happens before `pending` is written, so a refused request
    # can never lock the host.
    body = _body_bytes(recipe["signup"]["body"])

    # A replacement keeps the old entry until the new signup returns a usable
    # key; every failure below puts it back untouched.
    previous = entry if entry and entry.get("state") == "active" else None
    pending = {"state": "pending", "recipe": recipe}
    if previous is not None:
        pending["previous"] = previous
    vault.hosts[host] = pending
    vault.save()  # pending is on disk before anything is sent

    try:
        status, _h, data = fetch(recipe["signup"]["method"], host, port, path,
                                 _headers(json_body=True), body)
    except NotSent as e:
        _abandon(vault, host, previous)
        return {"status": "error", "reason": str(e)}
    except Exception as e:  # timeout, reset, too large: it may have arrived
        log(f"[broker] register {host} failed: {type(e).__name__}")
        return _uncertain(vault, host, previous,
                          f"signup failed mid-request ({type(e).__name__}); the account may already exist")

    log(f"[broker] register {host} {path.split('?')[0]} -> {status}")
    if 400 <= status < 500:
        _abandon(vault, host, previous)
        return {"status": "error", "http_status": status,
                "body": redact(_decode(data), [], shape=True)}
    if not 200 <= status < 300:
        return _uncertain(vault, host, previous,
                          f"signup returned HTTP {status}; the account may already exist")
    try:
        parsed = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return _uncertain(vault, host, previous,
                          "signup response was not JSON; the account may already exist")

    entry = vault.hosts[host]
    entry["saved_response"] = parsed
    entry["saved_status"] = status
    vault.save()  # saved before reading anything from it
    return _finish(host, entry, vault, log)


def _finish(host: str, entry: dict, vault: Vault, log) -> dict:
    saved = entry["saved_response"]
    value = extract(saved, entry["recipe"]["secret"])
    if looks_like_key(value):
        # The saved response stays in the vault: if the agent named a wrong
        # field that merely looks like a key, the site will reject it and the
        # agent can swap in the real key from here. It is never returned or
        # sent anywhere.
        vault.hosts[host] = {
            "state": "active",
            "key": value,
            "recipe": entry["recipe"],
            "saved_response": saved,
            "saved_status": entry.get("saved_status"),
        }
        vault.save()
        log(f"[broker] register {host} stored a key")
        return {"status": "ok", "http_status": entry.get("saved_status"),
                "body": redact(saved, [value], shape=True)}

    entry["failed_paths"] = entry.get("failed_paths", 0) + 1
    if entry["failed_paths"] >= MAX_PATH_TRIES:
        return _needs_human(vault, host, f"no key found after {MAX_PATH_TRIES} secret paths")
    vault.save()
    return {
        "status": "error",
        "reason": "secret path not found, or its value doesn't look like a key; call register again with a corrected path",
        "fields": field_paths(saved),
        "tries_left": MAX_PATH_TRIES - entry["failed_paths"],
    }


def request(payload: dict, vault: Vault, fetch, log) -> dict:
    host, port, path = parse_target(payload.get("url"))
    method = str(payload.get("method") or "GET").upper()
    if method not in _REQUEST_METHODS:
        raise Refused("unsupported method")
    entry = vault.hosts.get(host)
    if not entry:
        raise Refused("no key for this host")
    if entry.get("state") != "active":
        raise Refused(f"this host is {entry.get('state')}, not ready")
    body = payload.get("body")
    if body is not None and method == "GET":
        raise Refused("GET cannot have a body")
    raw = _body_bytes(body)

    key = entry["key"]
    attach = entry["recipe"]["attach"]
    value = f"{attach['scheme']} {key}".strip()
    headers = _headers({attach["header"]: value}, json_body=raw is not None)
    try:
        status, rheaders, data = fetch(method, host, port, path, headers, raw)
    except (NotSent, Refused) as e:
        return {"status": "error", "reason": str(e)}
    except Exception as e:
        return {"status": "error", "reason": f"request failed ({type(e).__name__})"}

    log(f"[broker] request {method} {host} {path.split('?')[0]} -> {status}")
    out = {"status": "ok", "http_status": status,
           "body": redact(_decode(data), [key], shape=False)}
    if status == 401:
        # The site rejected the key: allow register again for this host
        # (a corrected secret path or header, or replace: true). A 403 only
        # means "not allowed to do this" and says nothing about the key.
        if not entry.get("key_rejected"):
            entry["key_rejected"] = True
            vault.save()
        out["note"] = ("the site rejected the key; you can call register again for "
                       "this host with a corrected secret path, or with replace: true")
    elif 200 <= status < 300 and entry.pop("key_rejected", None):
        vault.save()
    if 300 <= status < 400:
        out["note"] = "redirect not followed"
        if "location" in rheaders:
            out["location"] = redact(rheaders["location"], [key], shape=False)
    return out


def handle(payload: Any, vault: Vault, fetch=pinned_fetch, log=print) -> dict:
    if not isinstance(payload, dict):
        return {"status": "refused", "reason": "payload must be an object"}
    try:
        op = payload.get("op")
        if op == "register":
            return register(payload.get("recipe"), vault, fetch, log,
                            replace=payload.get("replace") is True)
        if op == "request":
            return request(payload, vault, fetch, log)
        raise Refused("op must be register or request")
    except Refused as e:
        return {"status": "refused", "reason": str(e)}
