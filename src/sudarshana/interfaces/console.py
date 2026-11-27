"""The console as a chat: owner messages are printed."""


class ConsoleNotifier:
    channel = "console"

    def send(self, text: str) -> None:
        print(f"\n[to owner]\n{text}\n")
