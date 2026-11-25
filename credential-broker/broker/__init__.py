"""Credential broker: holds outside-service keys so the agent never sees them."""

from .core import Vault, handle

__all__ = ["Vault", "handle"]
