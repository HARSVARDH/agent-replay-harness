"""Pytest fixtures for agent replay harness tests."""

from pathlib import Path
import tempfile
import pytest

from agent_replay_harness.storage.file import FileTapeStorage
from agent_replay_harness.storage.sqlite import SQLiteTapeStorage
from agent_replay_harness.observability import clear_captured_spans


@pytest.fixture
def temp_tape_dir():
    """Create a temporary directory for file tape storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def file_storage(temp_tape_dir):
    """FileTapeStorage instance pointing to temporary directory."""
    return FileTapeStorage(base_dir=temp_tape_dir)


@pytest.fixture
def sqlite_storage(temp_tape_dir):
    """SQLiteTapeStorage instance with isolated database."""
    return SQLiteTapeStorage(db_path=temp_tape_dir / "test_harness.db")


@pytest.fixture(autouse=True)
def reset_spans():
    """Clear OpenTelemetry in-memory spans before each test."""
    clear_captured_spans()
    yield
    clear_captured_spans()
