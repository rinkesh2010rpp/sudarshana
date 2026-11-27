"""CheckpointStore port on a SQLite file (LangGraph's SqliteSaver)."""


class SqliteCheckpoints:
    def __init__(self, path: str):
        self.path = path

    def open(self):
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(self.path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        conn.execute("PRAGMA journal_mode=DELETE")
        return checkpointer
