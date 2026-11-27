"""KVStore port on a named modal.Dict."""
import modal


def modal_kv(name: str) -> modal.Dict:
    """A named modal.Dict. It already has the KVStore methods; the handle is
    lazy, so nothing is fetched until first use."""
    return modal.Dict.from_name(name, create_if_missing=True)
