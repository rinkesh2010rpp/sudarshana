"""Shared test helpers.

Code moved out of agent/main.py into src/sudarshana/ imports names like
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
            if (modname == "main" or modname.startswith("sudarshana.")) and hasattr(m, name):
                monkeypatch.setattr(m, name, value)
                hit = True
        assert hit, f"{name} is not defined in any loaded module"

    return _patch
