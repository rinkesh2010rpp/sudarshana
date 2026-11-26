"""Sending a message to Rinkesh's Telegram chat."""
import os

def _send_telegram(text: str) -> None:
    """Send `text`, split into chunks under Telegram's ~4096-char message limit."""
    import requests

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_ALLOWED_USER_ID"]
    chunk_size = 3500
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]
    for chunk in chunks:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": chunk},
            timeout=10,
        )
