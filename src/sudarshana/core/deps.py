"""The real implementations of the ports, held in one place.

Built once per container by the runtime (infra/runtime/) and passed
explicitly. There is no global Deps.
"""
from dataclasses import dataclass

from sudarshana.core.ports import CheckpointStore, KVStore, Notifier, Storage


@dataclass
class Deps:
    storage: Storage
    checkpoints: CheckpointStore
    status_kv: KVStore
    claims_kv: KVStore
    a2a_kv: KVStore
    # Messages for the owner that no incoming message asked for: asks,
    # hourly reports, a2a alerts. A reply goes back where its message came in.
    owner_notifier: Notifier
