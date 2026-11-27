"""The Modal Volume that holds the agent's workspace."""
import modal

# Persistent disk for the agent's file tools and its VISION/ROADMAP/actions/
# logs hierarchy — without it, writes vanish at the end of each invocation.
volume = modal.Volume.from_name("sudarshana-files", create_if_missing=True)
