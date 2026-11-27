"""Shared test helpers.

Code in src/sudarshana/ imports names like
VOLUME_PATH, volume and ASKS_PATH from each other, so a value lives in every
module that imported it. `patch_all` sets one name to one value in all of
them at once, which is what patching `main` alone used to do.
"""
import sys

import pytest


@pytest.fixture()
def patch_all(monkeypatch):
    def _patch(name, value):
        hit = False
        for modname, m in list(sys.modules.items()):
            if modname.startswith("sudarshana.") and hasattr(m, name):
                monkeypatch.setattr(m, name, value)
                hit = True
        assert hit, f"{name} is not defined in any loaded module"

    return _patch


class FakeStorage:
    """Storage port: counts commits; `on_commit` sees each one."""

    def __init__(self, root="", on_commit=None):
        self.root = root
        self.commits = 0
        self.on_commit = on_commit

    def commit(self):
        self.commits += 1
        if self.on_commit:
            self.on_commit()

    def reload(self):
        pass


class FakeNotifier:
    """Notifier port: keeps what was sent; `send` can be swapped to fail."""

    channel = "test chat"

    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)


def fake_deps(kv, storage=None, owner_notifier=None):
    """A Deps of fakes. `kv(name)` builds each named key-value store."""
    from sudarshana.core.deps import Deps

    return Deps(
        storage=storage or FakeStorage(),
        checkpoints=None,
        status_kv=kv("sudarshana-status"),
        claims_kv=kv("sudarshana-telegram-claims"),
        a2a_kv=kv("sudarshana-a2a-agents"),
        owner_notifier=owner_notifier or FakeNotifier(),
    )
