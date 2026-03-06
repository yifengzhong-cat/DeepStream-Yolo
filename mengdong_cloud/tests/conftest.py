import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ["MENGDONG_RUNTIME_DIR"] = "/tmp/mengdong_cloud_tests"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(app)
