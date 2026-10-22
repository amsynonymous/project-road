import os
import importlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture()
def app(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("ROAD_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("ROAD_MODEL_PATH", str(tmp_path / "missing_best.pt"))
    import app as road_app
    road_app.DB_PATH = db_path
    road_app.WEIGHTS_PATH = tmp_path / "missing_best.pt"
    road_app.INSTANCE_DIR = tmp_path
    road_app.UPLOAD_DIR = tmp_path / "uploads"
    road_app.RESULT_DIR = tmp_path / "results"
    road_app.FRAME_DIR = tmp_path / "frames"
    for folder in (road_app.UPLOAD_DIR, road_app.RESULT_DIR, road_app.FRAME_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    road_app.init_db()
    road_app.app.config.update(TESTING=True, SECRET_KEY="test-secret")
    yield road_app.app
    road_app._model = None

@pytest.fixture()
def client(app):
    return app.test_client()
