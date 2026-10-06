"""SQLite-backed tape storage engine for relational and indexed event persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from agent_replay_harness.hashing import canonical_json_dumps
from agent_replay_harness.schemas import (
    AgentRunEvent,
    CassetteManifest,
    CassetteTape,
    ExecutionMode,
    RunEventType,
    RunSummary,
    StateSnapshot,
    ToolCallRecord,
)
from agent_replay_harness.storage.base import BaseTapeStorage


class SQLiteTapeStorage(BaseTapeStorage):
    """Stores tapes, events, snapshots, and tool cassettes in SQLite."""

    def __init__(self, db_path: str | Path = ".tapes/harness.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    parent_run_id TEXT,
                    fork_step INTEGER,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_steps INTEGER DEFAULT 0,
                    total_tool_calls INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    error TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parent_run_id TEXT,
                    fork_step INTEGER,
                    step INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    trace_id TEXT,
                    span_id TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    state_data_json TEXT NOT NULL,
                    parent_snapshot_id TEXT,
                    timestamp TEXT NOT NULL,
                    diff_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    args_json TEXT NOT NULL,
                    kwargs_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    duration_ms REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    hash_key TEXT NOT NULL,
                    call_index INTEGER DEFAULT 0,
                    trace_id TEXT,
                    span_id TEXT,
                    replayed INTEGER DEFAULT 0,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_run_step ON events (run_id, step)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snaps_run_step ON snapshots (run_id, step)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_run_hash ON tool_calls (run_id, hash_key, call_index)")
            conn.commit()

    def create_run(self, manifest: CassetteManifest) -> None:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, parent_run_id, fork_step, mode, status,
                    created_at, completed_at, total_steps, total_tool_calls,
                    metadata_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.run_id,
                    manifest.parent_run_id,
                    manifest.fork_step,
                    manifest.mode.value,
                    manifest.status,
                    manifest.created_at.isoformat(),
                    manifest.completed_at.isoformat() if manifest.completed_at else None,
                    manifest.total_steps,
                    manifest.total_tool_calls,
                    json.dumps(manifest.metadata),
                    manifest.error,
                ),
            )
            conn.commit()

    def update_run_manifest(self, manifest: CassetteManifest) -> None:
        self.create_run(manifest)

    def get_run_manifest(self, run_id: str) -> Optional[CassetteManifest]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return CassetteManifest(
                run_id=row["run_id"],
                parent_run_id=row["parent_run_id"],
                fork_step=row["fork_step"],
                mode=ExecutionMode(row["mode"]),
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                total_steps=row["total_steps"],
                total_tool_calls=row["total_tool_calls"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                error=row["error"],
            )

    def append_event(self, event: AgentRunEvent) -> None:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, run_id, parent_run_id, fork_step, step,
                    event_type, payload_json, timestamp, trace_id, span_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.parent_run_id,
                    event.fork_step,
                    event.step,
                    event.event_type.value,
                    json.dumps(event.payload, default=str),
                    event.timestamp.isoformat(),
                    event.trace_id,
                    event.span_id,
                ),
            )
            conn.commit()

    def get_events(self, run_id: str) -> List[AgentRunEvent]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY step ASC, timestamp ASC",
                (run_id,),
            )
            rows = cursor.fetchall()
            return [
                AgentRunEvent(
                    event_id=row["event_id"],
                    run_id=row["run_id"],
                    parent_run_id=row["parent_run_id"],
                    fork_step=row["fork_step"],
                    step=row["step"],
                    event_type=RunEventType(row["event_type"]),
                    payload=json.loads(row["payload_json"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    trace_id=row["trace_id"],
                    span_id=row["span_id"],
                )
                for row in rows
            ]

    def save_snapshot(self, snapshot: StateSnapshot) -> None:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO snapshots (
                    snapshot_id, run_id, step, node_name, state_data_json,
                    parent_snapshot_id, timestamp, diff_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.run_id,
                    snapshot.step,
                    snapshot.node_name,
                    canonical_json_dumps(snapshot.state_data),
                    snapshot.parent_snapshot_id,
                    snapshot.timestamp.isoformat(),
                    json.dumps(snapshot.diff_from_parent) if snapshot.diff_from_parent else None,
                ),
            )
            conn.commit()

    def get_snapshot(self, run_id: str, step: int) -> Optional[StateSnapshot]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM snapshots WHERE run_id = ? AND step = ?",
                (run_id, step),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return StateSnapshot(
                snapshot_id=row["snapshot_id"],
                run_id=row["run_id"],
                step=row["step"],
                node_name=row["node_name"],
                state_data=json.loads(row["state_data_json"]),
                parent_snapshot_id=row["parent_snapshot_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                diff_from_parent=json.loads(row["diff_json"]) if row["diff_json"] else None,
            )

    def list_snapshots(self, run_id: str) -> List[StateSnapshot]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM snapshots WHERE run_id = ? ORDER BY step ASC",
                (run_id,),
            )
            rows = cursor.fetchall()
            return [
                StateSnapshot(
                    snapshot_id=row["snapshot_id"],
                    run_id=row["run_id"],
                    step=row["step"],
                    node_name=row["node_name"],
                    state_data=json.loads(row["state_data_json"]),
                    parent_snapshot_id=row["parent_snapshot_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    diff_from_parent=json.loads(row["diff_json"]) if row["diff_json"] else None,
                )
                for row in rows
            ]

    def record_tool_call(self, record: ToolCallRecord) -> None:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO tool_calls (
                    call_id, run_id, step, tool_name, args_json, kwargs_json,
                    result_json, error, duration_ms, timestamp, hash_key,
                    call_index, trace_id, span_id, replayed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.call_id,
                    record.run_id,
                    record.step,
                    record.tool_name,
                    canonical_json_dumps(record.args),
                    canonical_json_dumps(record.kwargs),
                    canonical_json_dumps(record.result) if record.result is not None else None,
                    record.error,
                    record.duration_ms,
                    record.timestamp.isoformat(),
                    record.hash_key,
                    record.call_index,
                    record.trace_id,
                    record.span_id,
                    1 if record.replayed else 0,
                ),
            )
            conn.commit()

    def get_tool_call(
        self,
        run_id: str,
        hash_key: str,
        call_index: int = 0,
    ) -> Optional[ToolCallRecord]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM tool_calls
                WHERE run_id = ? AND hash_key = ? AND call_index = ?
                """,
                (run_id, hash_key, call_index),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return ToolCallRecord(
                call_id=row["call_id"],
                run_id=row["run_id"],
                step=row["step"],
                tool_name=row["tool_name"],
                args=json.loads(row["args_json"]),
                kwargs=json.loads(row["kwargs_json"]),
                result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
                error=row["error"],
                duration_ms=row["duration_ms"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                hash_key=row["hash_key"],
                call_index=row["call_index"],
                trace_id=row["trace_id"],
                span_id=row["span_id"],
                replayed=bool(row["replayed"]),
            )

    def list_tool_calls(self, run_id: str) -> List[ToolCallRecord]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY step ASC, timestamp ASC",
                (run_id,),
            )
            rows = cursor.fetchall()
            return [
                ToolCallRecord(
                    call_id=row["call_id"],
                    run_id=row["run_id"],
                    step=row["step"],
                    tool_name=row["tool_name"],
                    args=json.loads(row["args_json"]),
                    kwargs=json.loads(row["kwargs_json"]),
                    result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
                    error=row["error"],
                    duration_ms=row["duration_ms"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    hash_key=row["hash_key"],
                    call_index=row["call_index"],
                    trace_id=row["trace_id"],
                    span_id=row["span_id"],
                    replayed=bool(row["replayed"]),
                )
                for row in rows
            ]

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
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [
                RunSummary(
                    run_id=row["run_id"],
                    parent_run_id=row["parent_run_id"],
                    fork_step=row["fork_step"],
                    mode=ExecutionMode(row["mode"]),
                    status=row["status"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    step_count=row["total_steps"],
                    tool_call_count=row["total_tool_calls"],
                    error=row["error"],
                )
                for row in rows
            ]

    def delete_run(self, run_id: str) -> bool:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
