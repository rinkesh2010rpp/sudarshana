"""
Point Telegram at your deployed (or `modal serve`-d) webhook URL.

Usage:
    python scripts/set_webhook.py https://your-modal-url
"""

import sys

import requests
from dotenv import load_dotenv
import os

load_dotenv()

if len(sys.argv) != 2:
    print("Usage: python scripts/set_webhook.py <webhook-url>")
    sys.exit(1)

url = sys.argv[1]
token = os.environ["TELEGRAM_BOT_TOKEN"]
secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()

body = {"url": url}
if secret:
    # Telegram stamps this value into the X-Telegram-Bot-Api-Secret-Token header
    # on every genuine delivery; the webhook rejects anything that doesn't match.
    body["secret_token"] = secret
else:
    print(
        "WARNING: TELEGRAM_WEBHOOK_SECRET is unset. The webhook will refuse every "
        "update (503) until you set it here and re-run this script. Set a random "
        "value in .env first."
    )

resp = requests.post(
    f"https://api.telegram.org/bot{token}/setWebhook",
    json=body,
    timeout=10,
)
print(resp.status_code, resp.json())
