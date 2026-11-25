"""Credential broker on Modal. Deploy into its OWN workspace, from your machine:

    modal deploy app.py

The agent calls the endpoint with a Modal proxy token for this workspace. It
must never get a Modal API token for this workspace.
"""

import modal

app = modal.App("credential-broker")

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "urllib3>=2,<3", "certifi")
    .add_local_python_source("broker")
)

vault_volume = modal.Volume.from_name("credential-broker-vault", create_if_missing=True)
VAULT_PATH = "/vault/vault.json"


@app.function(
    image=image,
    volumes={"/vault": vault_volume},
    # One container, one request at a time: vault writes can't collide.
    max_containers=1,
    timeout=90,
)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def broker(payload: dict) -> dict:
    from broker import Vault, handle

    vault_volume.reload()  # pick up a human's edit to the vault file
    vault = Vault(VAULT_PATH, commit=vault_volume.commit)
    return handle(payload, vault)
