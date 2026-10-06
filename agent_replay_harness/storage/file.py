"""File-based tape storage engine using JSON manifests and append-only JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading
from typing import Any, Dict, List, Optional

from agent_replay_harness.hashing import canonical_json_dumps
from agent_replay_harness.schemas import (
    AgentRunEvent,
    CassetteManifest,
    CassetteTape,
    RunSummary,
    StateSnapshot,
    ToolCallRecord,
)
from agent_replay_harness.storage.base import BaseTapeStorage


class FileTapeStorage(BaseTapeStorage):
    """Stores tapes as readable JSON manifests and JSONL files on disk."""

    def __init__(self, base_dir: str | Path = ".tapes") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _run_dir(self, run_id: str) -> Path:
        run_path = self.base_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path

    def _manifest_file(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "manifest.json"

    def _events_file(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "events.jsonl"

    def _snapshots_file(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "snapshots.jsonl"

    def _cassette_file(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "cassette.json"

    def create_run(self, manifest: CassetteManifest) -> None:
        with self._lock:
            m_path = self._manifest_file(manifest.run_id)
            with open(m_path, "w", encoding="utf-8") as f:
                f.write(manifest.model_dump_json(indent=2))

    def update_run_manifest(self, manifest: CassetteManifest) -> None:
        with self._lock:
            m_path = self._manifest_file(manifest.run_id)
            with open(m_path, "w", encoding="utf-8") as f:
                f.write(manifest.model_dump_json(indent=2))

    def get_run_manifest(self, run_id: str) -> Optional[CassetteManifest]:
        m_path = self._manifest_file(run_id)
        if not m_path.exists():
            return None
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CassetteManifest.model_validate(data)

    def append_event(self, event: AgentRunEvent) -> None:
        with self._lock:
            e_path = self._events_file(event.run_id)
            with open(e_path, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")

    def get_events(self, run_id: str) -> List[AgentRunEvent]:
        e_path = self._events_file(run_id)
        if not e_path.exists():
            return []
        events: List[AgentRunEvent] = []
        with open(e_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(AgentRunEvent.model_validate_json(line))
        return events

    def save_snapshot(self, snapshot: StateSnapshot) -> None:
        with self._lock:
            s_path = self._snapshots_file(snapshot.run_id)
            with open(s_path, "a", encoding="utf-8") as f:
                f.write(snapshot.model_dump_json() + "\n")

    def get_snapshot(self, run_id: str, step: int) -> Optional[StateSnapshot]:
        snaps = self.list_snapshots(run_id)
        for s in snaps:
            if s.step == step:
                return s
        return None

    def list_snapshots(self, run_id: str) -> List[StateSnapshot]:
        s_path = self._snapshots_file(run_id)
        if not s_path.exists():
            return []
        snaps: List[StateSnapshot] = []
        with open(s_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    snaps.append(StateSnapshot.model_validate_json(line))
        snaps.sort(key=lambda s: s.step)
        return snaps

    def record_tool_call(self, record: ToolCallRecord) -> None:
        with self._lock:
            c_path = self._cassette_file(record.run_id)
            records: List[Dict[str, Any]] = []
            if c_path.exists():
                try:
                    with open(c_path, "r", encoding="utf-8") as f:
                        records = json.load(f)
                except Exception:
                    records = []
            records.append(record.model_dump(mode="json"))
            with open(c_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

    def get_tool_call(
        self,
        run_id: str,
        hash_key: str,
        call_index: int = 0,
    ) -> Optional[ToolCallRecord]:
        calls = self.list_tool_calls(run_id)
        matching = [c for c in calls if c.hash_key == hash_key]
        if 0 <= call_index < len(matching):
            return matching[call_index]
        return None

    def list_tool_calls(self, run_id: str) -> List[ToolCallRecord]:
        c_path = self._cassette_file(run_id)
        if not c_path.exists():
            return []
        with open(c_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                return []
        return [ToolCallRecord.model_validate(item) for item in data]

    def get_tape(self, run_id: str) -> Optional[CassetteTape]:
        manifest = self.get_run_manifest(run_id)
        if not manifest:
            return None
        events = self.get_events(run_id)
        snapshots = self.list_snapshots(run_id)
        tool_calls = self.list_tool_calls(run_id)
        return CassetteTape(
            manifest=manifest,
            events=events,
            snapshots=snapshots,
            tool_calls=tool_calls,
        )

    def list_runs(self) -> List[RunSummary]:
        summaries: List[RunSummary] = []
        if not self.base_dir.exists():
            return summaries

        for entry in self.base_dir.iterdir():
            if entry.is_dir():
                m_file = entry / "manifest.json"
                if m_file.exists():
                    try:
                        manifest = self.get_run_manifest(entry.name)
                        if manifest:
                            summaries.append(
                                RunSummary(
                                    run_id=manifest.run_id,
                                    parent_run_id=manifest.parent_run_id,
                                    fork_step=manifest.fork_step,
                                    mode=manifest.mode,
                                    status=manifest.status,
                                    created_at=manifest.created_at,
                                    step_count=manifest.total_steps,
                                    tool_call_count=manifest.total_tool_calls,
                                    error=manifest.error,
                                )
                            )
                    except Exception:
                        continue

        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            run_dir = self.base_dir / run_id
            if run_dir.exists() and run_dir.is_dir():
                shutil.rmtree(run_dir)
                return True
            return False
