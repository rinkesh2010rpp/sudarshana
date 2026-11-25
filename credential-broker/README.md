# credential-broker

Lets the agent register for and use outside services without holding their
keys. The agent sends a JSON request to this endpoint; the broker signs up,
keeps the key in its vault, and attaches it only to that exact host. Nothing it
returns contains a key.

- `broker/core.py`: the logic (vault, target checks, redaction, `register`, `request`). No Modal.
- `app.py`: the Modal endpoint. One container, proxy-token auth, vault on a Volume.
- `tests/`: runs against a fake site. `python -m pytest -q credential-broker/tests`

v1 handles static API keys only. A registered host accepts only GET until a
human allows other methods in the vault.

## Where it runs

For now it deploys into the agent's own Modal workspace. That is a known gap:
the agent's shell can open the vault Volume by name and read stored keys. The
plan is to move it to its own workspace (or a Cloudflare Worker); only `app.py`
and the deploy profile change.

Deploy only after this is merged:

    cd credential-broker
    $env:PYTHONUTF8="1"; modal deploy app.py

The agent calls it with a Modal proxy token (`modal workspace proxy-tokens create`).

## Allowing more than GET for a host

The vault is `/vault.json` on the `credential-broker-vault` Volume. Download it,
add the method to that host's `allowed_methods`, and upload it:

    modal volume get credential-broker-vault vault.json .
    modal volume put credential-broker-vault vault.json /vault.json --force
