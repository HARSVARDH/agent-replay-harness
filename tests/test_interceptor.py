"""Unit tests for the VCR tool interceptor and @record_tool_call decorator."""

import pytest

from agent_replay_harness.context import execution_scope
from agent_replay_harness.interceptor import (
    CassetteNotFoundError,
    record_tool_call,
)
from agent_replay_harness.schemas import ExecutionMode


def test_interceptor_record_and_replay_mode(file_storage):
    call_counter = 0

    @record_tool_call("mock_api_call")
    def mock_api_call(param: str) -> dict:
        nonlocal call_counter
        call_counter += 1
        return {"param": param, "counter": call_counter}

    # 1. Record Mode
    with execution_scope(run_id="run-rec-1", mode=ExecutionMode.RECORD, storage=file_storage):
        res1 = mock_api_call("first")
        assert res1["counter"] == 1
        assert call_counter == 1

    # Verify stored in cassette
    tool_calls = file_storage.list_tool_calls("run-rec-1")
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "mock_api_call"
    assert tool_calls[0].result["counter"] == 1

    # 2. Replay Mode: underlying function must NOT be invoked
    with execution_scope(
        run_id="replay-1",
        mode=ExecutionMode.REPLAY,
        parent_run_id="run-rec-1",
        storage=file_storage,
    ):
        res_replay = mock_api_call("first")
        assert res_replay["counter"] == 1
        # Crucial check: call_counter did NOT increase!
        assert call_counter == 1


def test_interceptor_sequential_calls(file_storage):
    """Verify that multiple invocations with identical parameters are matched in order."""
    call_seq = 0

    @record_tool_call("dice_roll")
    def dice_roll() -> int:
        nonlocal call_seq
        call_seq += 10
        return call_seq

    with execution_scope(run_id="dice-run", mode=ExecutionMode.RECORD, storage=file_storage):
        r1 = dice_roll()  # 10
        r2 = dice_roll()  # 20
        assert r1 == 10
        assert r2 == 20

    # Replay sequentially
    with execution_scope(
        run_id="dice-replay",
        mode=ExecutionMode.REPLAY,
        parent_run_id="dice-run",
        storage=file_storage,
    ):
        rep1 = dice_roll()
        rep2 = dice_roll()
        assert rep1 == 10
        assert rep2 == 20


def test_interceptor_missing_cassette_error(file_storage):
    @record_tool_call("missing_tool")
    def missing_tool():
        return 42

    with execution_scope(
        run_id="fresh-replay",
        mode=ExecutionMode.REPLAY,
        storage=file_storage,
        strict_replay=True,
    ):
        with pytest.raises(CassetteNotFoundError):
            missing_tool()


def test_interceptor_fault_injection(file_storage):
    """Verify patch_rules override live/cassette execution to simulate errors."""
    @record_tool_call("external_service")
    def external_service() -> str:
        return "normal response"

    with execution_scope(
        run_id="fault-run",
        mode=ExecutionMode.FORK,
        storage=file_storage,
        patch_rules={"tool_result": "Injected Mock 500"},
    ):
        res = external_service()
        assert res == "Injected Mock 500"
