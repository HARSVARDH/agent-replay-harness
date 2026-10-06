"""Tests for production crash-camera bundle export and import."""

from pathlib import Path
import tempfile

from agent_replay_harness.crash_cam import export_crash_bundle, import_crash_bundle
from agent_replay_harness.runner import run_agent
from agent_replay_harness.schemas import ExecutionMode
from agent_replay_harness.storage.file import FileTapeStorage


def test_export_and_import_crash_bundle(file_storage):
    # 1. Run live agent
    res = run_agent(query="Analyze NVDA risk", mode=ExecutionMode.RECORD, storage=file_storage)
    run_id = res["run_id"]

    # 2. Export crash bundle to temp file
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = Path(tmp_dir) / "incident.harness-crash"
        exported = export_crash_bundle(run_id=run_id, output_path=bundle_path, storage=file_storage)
        assert exported.exists()

        # 3. Import bundle into an entirely isolated new storage
        new_storage_dir = Path(tmp_dir) / "fresh_storage"
        fresh_storage = FileTapeStorage(new_storage_dir)

        imported_id = import_crash_bundle(bundle_path=bundle_path, target_storage=fresh_storage)
        assert imported_id == run_id

        # Verify all snapshots and cassettes restored
        manifest = fresh_storage.get_run_manifest(imported_id)
        assert manifest is not None
        assert manifest.run_id == run_id

        snaps = fresh_storage.list_snapshots(imported_id)
        assert len(snaps) >= 4

        tools = fresh_storage.list_tool_calls(imported_id)
        assert len(tools) >= 1
