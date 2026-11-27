"""The Modal Volume that holds the agent's workspace."""
import modal

# Persistent disk for the agent's file tools and its VISION/ROADMAP/actions/
# logs hierarchy — without it, writes vanish at the end of each invocation.
volume = modal.Volume.from_name("sudarshana-files", create_if_missing=True)


class ModalStorage:
    """Storage port on a Modal Volume mounted at `root`."""

    def __init__(self, vol: modal.Volume, root: str):
        self._vol = vol
        self.root = root

    def commit(self) -> None:
        self._vol.commit()

    def reload(self) -> None:
        self._vol.reload()
