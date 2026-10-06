"""Integration tests verifying deterministic replay of LangGraph workflows."""

from agent_replay_harness.runner import replay_agent, run_agent
from agent_replay_harness.schemas import ExecutionMode


def test_end_to_end_record_and_deterministic_replay(file_storage):
    # 1. Record live workflow run
    run_result = run_agent(
        query="Analyze NVDA financial risk",
        mode=ExecutionMode.RECORD,
        storage=file_storage,
    )

    run_id = run_result["run_id"]
    output_1 = run_result["output"]
    assert output_1["ticker"] == "NVDA"
    assert output_1["status"] == "completed"
    assert "market_data" in output_1
    assert "risk_assessment" in output_1

    # Verify cassette records
    tool_calls = file_storage.list_tool_calls(run_id)
    assert len(tool_calls) == 4  # quote, filing, risk model, report

    # 2. Replay run deterministically
    replay_result = replay_agent(
        run_id=run_id,
        storage=file_storage,
        strict=True,
    )

    output_2 = replay_result["output"]

    # Exact deterministic equivalence
    assert output_1["market_data"]["price"] == output_2["market_data"]["price"]
    assert output_1["market_data"]["timestamp"] == output_2["market_data"]["timestamp"]
    assert output_1["risk_assessment"]["value_at_risk_95"] == output_2["risk_assessment"]["value_at_risk_95"]
    assert output_1["final_report"]["publishing_receipt"]["report_id"] == output_2["final_report"]["publishing_receipt"]["report_id"]
    assert replay_result["total_tool_calls"] == 4
