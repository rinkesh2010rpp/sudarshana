"""Where the agent's workspace disk is mounted."""
import os

# /data on Modal. SUDARSHANA_DATA points it elsewhere for the server runtime
# (infra/runtime/server); it is never set in the Modal app.
VOLUME_PATH = os.environ.get("SUDARSHANA_DATA", "/data")
