"""Unit tests for schemas, hashing, and dict diffing."""

from datetime import datetime, timezone

from agent_replay_harness.hashing import (
    canonical_json_dumps,
    compute_dict_diff,
    hash_tool_call,
)
from agent_replay_harness.schemas import (
    AgentRunEvent,
    CassetteManifest,
    ExecutionMode,
    RunEventType,
    StateSnapshot,
    ToolCallRecord,
)


def test_tool_call_record_serialization():
    record = ToolCallRecord(
        run_id="run-123",
        step=2,
        tool_name="fetch_quote",
        args=["TSLA"],
        kwargs={"timeout": 10},
        result={"price": 250.0},
        hash_key="abc12345",
    )
    raw_json = record.model_dump_json()
    loaded = ToolCallRecord.model_validate_json(raw_json)
    assert loaded.run_id == "run-123"
    assert loaded.tool_name == "fetch_quote"
    assert loaded.result["price"] == 250.0


def test_hash_tool_call_determinism():
    h1 = hash_tool_call("fetch_data", ["AAPL", 100], {"currency": "USD", "detailed": True})
    # Swap key order in kwargs
    h2 = hash_tool_call("fetch_data", ["AAPL", 100], {"detailed": True, "currency": "USD"})
    assert h1 == h2
    assert len(h1) == 16


def test_compute_dict_diff():
    old_state = {"a": 1, "b": "hello", "c": [1, 2]}
    new_state = {"b": "world", "c": [1, 2], "d": True}

    diff = compute_dict_diff(old_state, new_state)
    assert diff["added"] == {"d": True}
    assert diff["removed"] == {"a": 1}
    assert diff["modified"] == {"b": {"old": "hello", "new": "world"}}


def test_agent_run_event_lifecycle():
    event = AgentRunEvent(
        run_id="run-1",
        step=1,
        event_type=RunEventType.RUN_STARTED,
        payload={"mode": ExecutionMode.RECORD.value},
    )
    assert event.event_type == RunEventType.RUN_STARTED
    assert event.payload["mode"] == "RECORD"
