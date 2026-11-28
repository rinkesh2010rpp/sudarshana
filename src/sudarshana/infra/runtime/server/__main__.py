"""`python -m sudarshana.infra.runtime.server` (from src/).

Loads .env (searched from the current folder upward) before anything from
the package is imported: some settings, like the tool screen switches, are
read once at import time.
"""
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from sudarshana.infra.runtime.server.app import main  # noqa: E402

main()
