"""What the agent needs from the outside world, with no say in how it's done.

Code outside interfaces/ and infra/ depends on these, never on modal,
fastapi or Telegram. infra/ and interfaces/ provide the real ones; tests
provide fakes.
"""
from typing import Any, Iterable, Protocol


class Storage(Protocol):
    """The agent's workspace disk."""

    root: str

    def commit(self) -> None:
        """Make writes so far durable and visible to other containers."""

    def reload(self) -> None:
        """Pick up writes other containers have committed."""


class KVStore(Protocol):
    """A small shared key-value store (status, dedup claims, a2a)."""

    def get(self, key: str, default: Any = None) -> Any: ...

    def put(self, key: str, value: Any, skip_if_exists: bool = False) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...

    def __delitem__(self, key: str) -> None: ...

    def keys(self) -> Iterable[str]: ...

    def items(self) -> Iterable[tuple[str, Any]]: ...


class CheckpointStore(Protocol):
    """Where the conversation thread is saved between turns."""

    def open(self) -> Any:
        """Return a ready LangGraph checkpointer."""


class Notifier(Protocol):
    """A way to send the owner a message."""

    def send(self, text: str) -> None: ...
