"""Integration tests for automated chaos engineering and resilience testing."""

from agent_replay_harness.chaos import run_chaos_matrix


def test_chaos_matrix_resilience_evaluation(file_storage):
    report = run_chaos_matrix(
        query="Analyze TSLA financial risk",
        storage=file_storage,
    )
    assert report.total_scenarios >= 4
    assert report.passed_scenarios == report.total_scenarios
    assert report.resilience_score_pct == 100.0

    for s in report.scenarios:
        assert s.recovered is True
        assert s.fork_run_id != "failed"
