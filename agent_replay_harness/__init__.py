"""Deterministic Replay & Time-Travel Debugging Harness for Agent Workflows."""

from agent_replay_harness.checkpointer import TapeCheckpointSaver
from agent_replay_harness.context import (
    ExecutionContext,
    execution_scope,
    get_current_context,
)
from agent_replay_harness.hashing import (
    canonical_json_dumps,
    compute_dict_diff,
    hash_tool_call,
)
from agent_replay_harness.interceptor import (
    CassetteNotFoundError,
    CassetteReplayError,
    record_tool_call,
)
from agent_replay_harness.runner import fork_agent, replay_agent, run_agent
from agent_replay_harness.schemas import (
    AgentRunEvent,
    CassetteManifest,
    CassetteTape,
    DiffResult,
    ExecutionMode,
    ForkRequest,
    RunEventType,
    RunSummary,
    StateSnapshot,
    ToolCallRecord,
)
from agent_replay_harness.storage.base import BaseTapeStorage
from agent_replay_harness.storage.file import FileTapeStorage
from agent_replay_harness.storage.sqlite import SQLiteTapeStorage

__version__ = "0.1.0"

__all__ = [
    "AgentRunEvent",
    "BaseTapeStorage",
    "CassetteManifest",
    "CassetteNotFoundError",
    "CassetteReplayError",
    "CassetteTape",
    "DiffResult",
    "ExecutionContext",
    "ExecutionMode",
    "FileTapeStorage",
    "ForkRequest",
    "RunEventType",
    "RunSummary",
    "SQLiteTapeStorage",
    "StateSnapshot",
    "TapeCheckpointSaver",
    "ToolCallRecord",
    "canonical_json_dumps",
    "compute_dict_diff",
    "execution_scope",
    "fork_agent",
    "get_current_context",
    "hash_tool_call",
    "record_tool_call",
    "replay_agent",
    "run_agent",
]
