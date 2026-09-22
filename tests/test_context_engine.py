import threading
from pathlib import Path

from nova_agent.core.context_engine import ContextEngine


def test_connection_can_be_used_from_the_worker_thread(tmp_path):
    # The connection opens on the main thread; commands run on the consumer
    # thread in --listen, which sqlite3's default thread check rejects.
    db_path = Path(tmp_path) / "nova.db"
    engine = ContextEngine(db_path)
    engine.set_alias("browser", "chrome")
    seen = {}

    def use_in_thread():
        seen["alias"] = engine.get_alias("browser")

    worker = threading.Thread(target=use_in_thread)
    worker.start()
    worker.join()

    assert seen["alias"] == "chrome"


def test_context_engine_records_and_reads_last_command(tmp_path):
    db_path = Path(tmp_path) / "nova.db"
    engine = ContextEngine(db_path)

    engine.log_command("open_app", "open chrome")

    assert engine.get_last_command() == ("open_app", "open chrome")


def test_context_engine_tracks_projects_and_preferences(tmp_path):
    db_path = Path(tmp_path) / "nova.db"
    engine = ContextEngine(db_path)

    engine.set_project("ml project", "C:/work/ml")
    engine.set_preference("default_project", "C:/work/ml")

    assert engine.get_project("ml project") == "C:/work/ml"
    assert engine.get_preference("default_project") == "C:/work/ml"


def test_context_engine_finds_project_from_partial_name(tmp_path):
    db_path = Path(tmp_path) / "nova.db"
    engine = ContextEngine(db_path)
    engine.set_project("ML Projects", "C:/work/ml")

    assert engine.find_project("ML Projects") == "C:/work/ml"
    assert engine.find_project("ml") == "C:/work/ml"
    assert engine.find_project("java") is None
    assert engine.find_project("") is None
