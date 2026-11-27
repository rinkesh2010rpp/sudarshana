"""Where the agent's workspace disk is mounted."""
import os

# /data on Modal. SUDARSHANA_DATA points it elsewhere for scripts/run_local.py;
# it is never set in the deployed app.
VOLUME_PATH = os.environ.get("SUDARSHANA_DATA", "/data")
