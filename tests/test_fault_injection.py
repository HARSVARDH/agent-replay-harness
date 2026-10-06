"""Integration tests for fault injection and automated recovery evaluation."""

from agent_replay_harness.runner import fork_agent, run_agent
from agent_replay_harness.schemas import ExecutionMode


def test_fault_injection_and_recovery_flow(file_storage):
    # 1. Base run
    orig = run_agent(
        query="Analyze TSLA risk and filings",
        mode=ExecutionMode.RECORD,
        storage=file_storage,
    )
    parent_id = orig["run_id"]

    # 2. Inject market circuit breaker fault into state at step 2
    forked = fork_agent(
        parent_run_id=parent_id,
        fork_at_step=3,
        patch_state={
            "market_data": {
                "status": "circuit_breaker_triggered",
                "error": "Flash crash circuit breaker active",
            }
        },
        storage=file_storage,
    )

    out = forked["output"]
    # Verify agent routed to recovery node
    assert "Remediation hedge executed" in out.get("action_taken", "")
    assert out["final_report"]["dossier"]["error_recovered"] == "Flash crash circuit breaker active"
    assert out["status"] == "completed"
