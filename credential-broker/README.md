# credential-broker

Lets the agent register for and use outside services without holding their
keys. The agent sends a JSON request to this endpoint; the broker signs up,
keeps the key in its vault, and attaches it only to that exact host. Nothing it
returns contains a key.

- `broker/core.py`: the logic (vault, target checks, redaction, `register`, `request`). No Modal.
- `app.py`: the Modal endpoint. One container, proxy-token auth, vault on a Volume.
- `tests/`: runs against a fake site. `python -m pytest -q credential-broker/tests`

v1 handles static API keys only. A registered host accepts GET, POST, PUT,
PATCH and DELETE; Taster's constitution check judges what the agent sends.

## Where it runs

For now it deploys into the agent's own Modal workspace. That is a known gap:
the agent's shell can open the vault Volume by name and read stored keys. The
plan is to move it to its own workspace (or a Cloudflare Worker); only `app.py`
and the deploy profile change.

Deploy only after this is merged:

    cd credential-broker
    $env:PYTHONUTF8="1"; modal deploy app.py

The agent calls it with a Modal proxy token (`modal workspace proxy-tokens create`).

The vault (`/vault.json` on the `credential-broker-vault` Volume) holds
plaintext keys. `vault.json` is in `.gitignore` so a stray copy can't be
committed.
