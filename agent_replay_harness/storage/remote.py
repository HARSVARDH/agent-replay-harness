"""Remote cloud tape storage client for syncing with the SaaS registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

from agent_replay_harness.schemas import (
    AgentRunEvent,
    CassetteManifest,
    CassetteTape,
    DiffResult,
    RunSummary,
    StateSnapshot,
    ToolCallRecord,
)
from agent_replay_harness.storage.base import BaseTapeStorage
from agent_replay_harness.storage.file import FileTapeStorage


class RemoteTapeStorage(BaseTapeStorage):
    """Hybrid storage client: caches locally while syncing to the centralized SaaS registry."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        local_cache_dir: str | Path = ".tapes",
    ) -> None:
        self.endpoint = (endpoint or os.environ.get("AGENT_HARNESS_ENDPOINT", "http://127.0.0.1:8000")).rstrip("/")
        self.api_key = api_key or os.environ.get("AGENT_HARNESS_API_KEY", "dev-secret-key")
        self.local_storage = FileTapeStorage(local_cache_dir)
        self.client = httpx.Client(
            base_url=self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=10.0,
        )

    def create_run(self, manifest: CassetteManifest) -> None:
        self.local_storage.create_run(manifest)
        try:
            self.client.post("/api/runs/execute", json={"query": manifest.metadata.get("query", ""), "mode": manifest.mode.value})
        except Exception:
            pass  # Local-first fallback

    def update_run_manifest(self, manifest: CassetteManifest) -> None:
        self.local_storage.update_run_manifest(manifest)

    def get_run_manifest(self, run_id: str) -> Optional[CassetteManifest]:
        local_m = self.local_storage.get_run_manifest(run_id)
        if local_m:
            return local_m
        try:
            res = self.client.get(f"/api/runs/{run_id}")
            if res.status_code == 200:
                manifest = CassetteManifest.model_validate(res.json())
                self.local_storage.create_run(manifest)
                return manifest
        except Exception:
            pass
        return None

    def append_event(self, event: AgentRunEvent) -> None:
        self.local_storage.append_event(event)

    def get_events(self, run_id: str) -> List[AgentRunEvent]:
        events = self.local_storage.get_events(run_id)
        if events:
            return events
        try:
            res = self.client.get(f"/api/runs/{run_id}/events")
            if res.status_code == 200:
                return [AgentRunEvent.model_validate(e) for e in res.json()]
        except Exception:
            pass
        return []

    def save_snapshot(self, snapshot: StateSnapshot) -> None:
        self.local_storage.save_snapshot(snapshot)

    def get_snapshot(self, run_id: str, step: int) -> Optional[StateSnapshot]:
        return self.local_storage.get_snapshot(run_id, step)

    def list_snapshots(self, run_id: str) -> List[StateSnapshot]:
        snaps = self.local_storage.list_snapshots(run_id)
        if snaps:
            return snaps
        try:
            res = self.client.get(f"/api/runs/{run_id}/snapshots")
            if res.status_code == 200:
                return [StateSnapshot.model_validate(s) for s in res.json()]
        except Exception:
            pass
        return []

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self.local_storage.record_tool_call(record)

    def get_tool_call(self, run_id: str, hash_key: str, call_index: int = 0) -> Optional[ToolCallRecord]:
        return self.local_storage.get_tool_call(run_id, hash_key, call_index)

    def list_tool_calls(self, run_id: str) -> List[ToolCallRecord]:
        calls = self.local_storage.list_tool_calls(run_id)
        if calls:
            return calls
        try:
            res = self.client.get(f"/api/runs/{run_id}/cassette")
            if res.status_code == 200:
                return [ToolCallRecord.model_validate(c) for c in res.json()]
        except Exception:
            pass
        return []

    def get_tape(self, run_id: str) -> Optional[CassetteTape]:
        tape = self.local_storage.get_tape(run_id)
        if tape:
            return tape
        manifest = self.get_run_manifest(run_id)
        if manifest:
            return CassetteTape(
                manifest=manifest,
                events=self.get_events(run_id),
                snapshots=self.list_snapshots(run_id),
                tool_calls=self.list_tool_calls(run_id),
            )
        return None

    def list_runs(self) -> List[RunSummary]:
        try:
            res = self.client.get("/api/runs")
            if res.status_code == 200:
                return [RunSummary.model_validate(r) for r in res.json()]
        except Exception:
            pass
        return self.local_storage.list_runs()

    def delete_run(self, run_id: str) -> bool:
        return self.local_storage.delete_run(run_id)

    def sync_tape_to_cloud(self, run_id: str) -> bool:
        """Push a local tape bundle to the SaaS cloud registry."""
        tape = self.local_storage.get_tape(run_id)
        if not tape:
            return False
        try:
            res = self.client.post("/api/registry/sync", json=tape.model_dump(mode="json"))
            return res.status_code in [200, 201]
        except Exception:
            return False
