"""Credential broker tools: outside-service calls through the broker."""
import os

# --- Credential broker (credential-broker/) ----------------------------------
# The broker holds outside-service keys; these tools only send it JSON. The
# proxy token is read in this Python process; the shell never gets it.
BROKER_TIMEOUT_S = 100  # above the broker function's own 90 s limit


def _broker_call(payload: dict) -> dict:
    import requests

    url = os.environ.get("BROKER_URL", "").strip()
    key = os.environ.get("BROKER_TOKEN_ID", "").strip()
    secret = os.environ.get("BROKER_TOKEN_SECRET", "").strip()
    if not (url and key and secret):
        return {"status": "error", "reason": "the credential broker is not configured "
                "(BROKER_URL, BROKER_TOKEN_ID, BROKER_TOKEN_SECRET)"}
    try:
        resp = requests.post(url, json=payload,
                             headers={"Modal-Key": key, "Modal-Secret": secret},
                             timeout=BROKER_TIMEOUT_S)
    except requests.RequestException as e:
        return {"status": "error", "reason": f"could not reach the broker ({type(e).__name__})"}
    if resp.status_code != 200:
        return {"status": "error", "reason": f"the broker answered HTTP {resp.status_code}"}
    try:
        return resp.json()
    except ValueError:
        return {"status": "error", "reason": "the broker returned non-JSON"}


def broker_register(recipe: dict, replace: bool = False) -> str:
    """Sign up for an outside service. The broker keeps the key; you never see it.

    recipe = {"signup": {"url": "https://...", "method": "POST", "body": {...}},
              "secret": "dotted.path.to.the.key.in.the.response",
              "attach": {"header": "Authorization", "scheme": "Bearer"}}
    Build it from the service's own docs. Returns the site's response with keys
    removed (it may hold a claim link for Rinkesh). If a later request gets a
    401, call this again with a corrected secret path or header; replace=True
    signs up fresh and may create a second account."""
    import json

    payload = {"op": "register", "recipe": recipe}
    if replace is True:
        payload["replace"] = True
    return json.dumps(_broker_call(payload))


def broker_request(url: str, method: str = "GET", body: dict | None = None) -> str:
    """Call a service you registered with through broker_register. The broker
    attaches the key for that exact host and returns the status and the
    response body with keys removed."""
    import json

    payload = {"op": "request", "url": url, "method": method}
    if body is not None:
        payload["body"] = body
    return json.dumps(_broker_call(payload))
