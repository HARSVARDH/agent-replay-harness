"""Integration tests for time-travel execution branching and state forking."""

from agent_replay_harness.runner import fork_agent, run_agent
from agent_replay_harness.schemas import ExecutionMode


def test_time_travel_fork_from_snapshot(file_storage):
    # 1. Base run
    orig_run = run_agent(
        query="Analyze AAPL risk and filings",
        mode=ExecutionMode.RECORD,
        storage=file_storage,
    )
    parent_id = orig_run["run_id"]

    # 2. Fork at step 3 with modified state
    fork_result = fork_agent(
        parent_run_id=parent_id,
        fork_at_step=3,
        patch_state={
            "market_data": {
                "ticker": "AAPL",
                "price": 999.99,
                "volume": 1000,
                "status": "normal",
            }
        },
        storage=file_storage,
    )

    fork_output = fork_result["output"]
    assert fork_result["parent_run_id"] == parent_id
    assert fork_result["fork_at_step"] == 3
    # Check that the forward execution computed risk using the patched price 999.99
    assert fork_output["market_data"]["price"] == 999.99
    assert fork_output["status"] == "completed"

    # Verify diff detected divergence
    diff = file_storage.diff_runs(parent_id, fork_result["run_id"])
    assert diff.diverged_at_step is not None
