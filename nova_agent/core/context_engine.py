import sqlite3
from pathlib import Path


class ContextEngine:
    def __init__(self, database_path: Path = Path("memory/nova.db")):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                name TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                last_opened TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS command_history (
                id INTEGER PRIMARY KEY,
                intent TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.connection.commit()

    def log_command(self, intent: str, raw_text: str) -> None:
        self.connection.execute(
            "INSERT INTO command_history (intent, raw_text) VALUES (?, ?)",
            (intent, raw_text),
        )
        self.connection.commit()

    def get_last_command(self) -> tuple[str, str] | None:
        row = self.connection.execute(
            "SELECT intent, raw_text FROM command_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row if row else None
