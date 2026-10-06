"""Pydantic v2 schemas for the Deterministic Replay & Time-Travel Debugging Harness.

Defines schemas for execution modes, run events, tool call records,
state snapshots, cassette manifests, and forking requests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field


class ExecutionMode(str, Enum):
    """Execution virtualization mode for agent loops and tools."""
    RECORD = "RECORD"          # Run live, record all tool calls and snapshots to tape
    REPLAY = "REPLAY"          # Deterministic execution served strictly from cassette tape
    PASSTHROUGH = "PASSTHROUGH"# Live execution without recording or mocking
    FORK = "FORK"              # Branch from step k with state/tool patches; replay 1..k-1, execute k..N


class RunEventType(str, Enum):
    """Categorization of agent trajectory events."""
    RUN_STARTED = "RUN_STARTED"
    NODE_ENTER = "NODE_ENTER"
    NODE_EXIT = "NODE_EXIT"
    TOOL_CALL = "TOOL_CALL"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"
    FORK_CREATED = "FORK_CREATED"


class ToolCallRecord(BaseModel):
    """VCR cassette entry for an intercepted tool call."""
    model_config = ConfigDict(extra="ignore")

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    step: int = 0
    tool_name: str
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hash_key: str
    call_index: int = 0
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    replayed: bool = False


class StateSnapshot(BaseModel):
    """Complete serialized memory footprint and scratchpad state at step k."""
    model_config = ConfigDict(extra="ignore")

    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    step: int
    node_name: str
    state_data: Dict[str, Any]
    parent_snapshot_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    diff_from_parent: Optional[Dict[str, Any]] = None


class AgentRunEvent(BaseModel):
    """Append-only event entry capturing transitions, tool payloads, and tokens."""
    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    parent_run_id: Optional[str] = None
    fork_step: Optional[int] = None
    step: int
    event_type: RunEventType
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class CassetteManifest(BaseModel):
    """Metadata manifest of a recorded run."""
    model_config = ConfigDict(extra="ignore")

    run_id: str
    parent_run_id: Optional[str] = None
    fork_step: Optional[int] = None
    mode: ExecutionMode
    status: str = "running"  # "running" | "completed" | "failed" | "forked"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_steps: int = 0
    total_tool_calls: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class CassetteTape(BaseModel):
    """Complete bundle containing manifest, events, snapshots, and tool cassettes."""
    model_config = ConfigDict(extra="ignore")

    manifest: CassetteManifest
    events: List[AgentRunEvent] = Field(default_factory=list)
    snapshots: List[StateSnapshot] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)


class RunSummary(BaseModel):
    """Lightweight summary of a run for listing and inspection."""
    model_config = ConfigDict(extra="ignore")

    run_id: str
    parent_run_id: Optional[str] = None
    fork_step: Optional[int] = None
    mode: ExecutionMode
    status: str
    created_at: datetime
    step_count: int = 0
    tool_call_count: int = 0
    error: Optional[str] = None


class ForkRequest(BaseModel):
    """Parameters for time-travel branching and state patching."""
    model_config = ConfigDict(extra="ignore")

    run_id: str
    fork_at_step: int
    patch_state: Optional[Dict[str, Any]] = None
    patch_tool_results: Optional[Dict[str, Any]] = None
    new_run_id: Optional[str] = None


class DiffResult(BaseModel):
    """Comparison result highlighting execution divergence between two runs."""
    model_config = ConfigDict(extra="ignore")

    run_1_id: str
    run_2_id: str
    diverged_at_step: Optional[int] = None
    step_comparisons: List[Dict[str, Any]] = Field(default_factory=list)
    tool_call_comparisons: List[Dict[str, Any]] = Field(default_factory=list)
