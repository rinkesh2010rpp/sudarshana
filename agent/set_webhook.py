"""
Point Telegram at your deployed (or `modal serve`-d) webhook URL.

Usage:
    python agent/set_webhook.py https://your-modal-url
"""

import sys

import requests
from dotenv import load_dotenv
import os

load_dotenv()

if len(sys.argv) != 2:
    print("Usage: python agent/set_webhook.py <webhook-url>")
    sys.exit(1)

url = sys.argv[1]
token = os.environ["TELEGRAM_BOT_TOKEN"]

resp = requests.post(
    f"https://api.telegram.org/bot{token}/setWebhook",
    json={"url": url},
    timeout=10,
)
print(resp.status_code, resp.json())
