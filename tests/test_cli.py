"""Tests for the Typer developer CLI interface."""

import json
from typer.testing import CliRunner

from agent_replay_harness.cli import app

runner = CliRunner()


def test_cli_full_workflow(temp_tape_dir):
    dir_str = str(temp_tape_dir)

    # 1. Run live agent
    run_res = runner.invoke(app, ["run", "--query", "Analyze MSFT financials", "--storage-dir", dir_str, "--json"])
    assert run_res.exit_code == 0
    run_data = json.loads(run_res.stdout)
    run_id = run_data["run_id"]
    assert run_id

    # 2. Replay run deterministically
    rep_res = runner.invoke(app, ["replay", "--run-id", run_id, "--storage-dir", dir_str, "--json"])
    assert rep_res.exit_code == 0
    rep_data = json.loads(rep_res.stdout)
    assert rep_data["replayed"] is True

    # 3. Fork via replay command with patch (as specified in checklist)
    fork_res = runner.invoke(
        app,
        [
            "replay",
            "--run-id", run_id,
            "--fork-at-step", "3",
            "--patch", '{"tool_result": "Error 404: Service Unavailable"}',
            "--storage-dir", dir_str,
            "--json",
        ],
    )
    assert fork_res.exit_code == 0
    fork_data = json.loads(fork_res.stdout)
    fork_id = fork_data["run_id"]
    assert fork_id

    # 4. List runs
    list_res = runner.invoke(app, ["list", "--storage-dir", dir_str])
    assert list_res.exit_code == 0
    assert "Recorded Agent Runs" in list_res.stdout

    # 5. Inspect run
    insp_res = runner.invoke(app, ["inspect", "--run-id", run_id, "--storage-dir", dir_str])
    assert insp_res.exit_code == 0
    assert run_id in insp_res.stdout

    # 6. Diff runs
    diff_res = runner.invoke(app, ["diff", "--run-1", run_id, "--run-2", fork_id, "--storage-dir", dir_str])
    assert diff_res.exit_code == 0
    assert "Execution Trajectory Diff" in diff_res.stdout
