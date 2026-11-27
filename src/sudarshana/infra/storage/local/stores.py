"""Storage and KVStore ports in this process: for run_local.py and tests."""


class LocalStorage:
    """A plain folder. Writes are durable as soon as they're made, so commit
    and reload have nothing to do."""

    def __init__(self, root: str):
        self.root = root

    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass


class DictKV:
    """A KVStore in a dict; gone when the process ends."""

    def __init__(self):
        self._d = {}

    def get(self, key, default=None):
        return self._d.get(key, default)

    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self._d:
            return False
        self._d[key] = value
        return True

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

    def keys(self):
        return list(self._d.keys())

    def items(self):
        return list(self._d.items())
