import sqlite3
from pathlib import Path


class ContextEngine:
    def __init__(self, database_path: Path = Path("memory/nova.db")):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        # Commands execute on the consumer thread while the connection opens on
        # the main thread, which sqlite3's default thread check rejects
        # (get_alias crashed mid-command). Usage is effectively sequential —
        # the main thread touches context only before the worker starts — so
        # one shared connection with the check relaxed is safe.
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
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
            CREATE TABLE IF NOT EXISTS app_aliases (
                alias TEXT PRIMARY KEY,
                app_key TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                fire_at INTEGER NOT NULL,
                fired INTEGER NOT NULL DEFAULT 0
            );
        """)
        self.connection.commit()

    def set_alias(self, alias: str, app_key: str) -> None:
        """Remember "browser" means "chrome" (or whatever the user prefers)."""
        self.connection.execute(
            "INSERT INTO app_aliases (alias, app_key) VALUES (?, ?) "
            "ON CONFLICT(alias) DO UPDATE SET app_key = excluded.app_key",
            (alias.strip().lower(), app_key),
        )
        self.connection.commit()

    def get_alias(self, alias: str) -> str | None:
        row = self.connection.execute(
            "SELECT app_key FROM app_aliases WHERE alias = ?",
            (alias.strip().lower(),),
        ).fetchone()
        return row[0] if row else None

    def list_aliases(self) -> list[tuple[str, str]]:
        """Every remembered alias as ``(alias, app_key)``.

        Used to build the STT name vocabulary: an alias the user taught
        ("browser" -> edge) is a word the model should expect to hear.
        """
        return self.connection.execute("SELECT alias, app_key FROM app_aliases").fetchall()

    def resolve_app(self, name: str) -> str:
        """Map a spoken app name onto a configured executable key."""
        candidate = (name or "").strip().lower()
        return self.get_alias(candidate) or candidate or "chrome"

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

    def set_project(self, name: str, path: str) -> None:
        self.connection.execute(
            "INSERT INTO projects (name, path) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET path = excluded.path, last_opened = CURRENT_TIMESTAMP",
            (name, path),
        )
        self.connection.commit()

    def get_project(self, name: str) -> str | None:
        row = self.connection.execute(
            "SELECT path FROM projects WHERE name = ?",
            (name,),
        ).fetchone()
        return row[0] if row else None

    def find_project(self, name: str) -> str | None:
        """Resolve a spoken project name, tolerating partial input.

        Voice input rarely matches the stored name exactly ("my ml folder" vs
        "ML Projects"), so fall back to a case-insensitive contains match.
        """
        candidate = name.strip()
        if not candidate:
            return None

        exact = self.get_project(candidate)
        if exact:
            return exact

        row = self.connection.execute(
            "SELECT path FROM projects "
            "WHERE lower(name) LIKE ? ORDER BY last_opened DESC LIMIT 1",
            (f"%{candidate.lower()}%",),
        ).fetchone()
        return row[0] if row else None

    def set_preference(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO preferences (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.connection.commit()

    def get_preference(self, key: str, default: str | None = None) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM preferences WHERE key = ?",
            (key,),
        ).fetchone()
        return row[0] if row else default

    def add_reminder(self, text: str, fire_at: int) -> int:
        """Store a reminder to speak at ``fire_at`` (unix seconds); returns its id."""
        cursor = self.connection.execute(
            "INSERT INTO reminders (text, fire_at) VALUES (?, ?)",
            (text.strip(), int(fire_at)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def due_reminders(self, now: int) -> list[tuple[int, str]]:
        """Unfired reminders whose time has come, oldest first: ``(id, text)``."""
        rows = self.connection.execute(
            "SELECT id, text FROM reminders "
            "WHERE fired = 0 AND fire_at <= ? ORDER BY fire_at",
            (int(now),),
        ).fetchall()
        return [(int(row[0]), row[1]) for row in rows]

    def mark_reminder_fired(self, reminder_id: int) -> None:
        self.connection.execute(
            "UPDATE reminders SET fired = 1 WHERE id = ?", (int(reminder_id),)
        )
        self.connection.commit()

    def pending_reminders(self, now: int) -> list[tuple[str, int]]:
        """Unfired reminders still in the future: ``(text, fire_at)``."""
        rows = self.connection.execute(
            "SELECT text, fire_at FROM reminders "
            "WHERE fired = 0 AND fire_at > ? ORDER BY fire_at",
            (int(now),),
        ).fetchall()
        return [(row[0], int(row[1])) for row in rows]

    def clear_reminders(self) -> int:
        """Drop every unfired reminder; returns how many were removed."""
        cursor = self.connection.execute("DELETE FROM reminders WHERE fired = 0")
        self.connection.commit()
        return int(cursor.rowcount)
