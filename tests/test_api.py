"""Tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient

from agent_replay_harness.server.app import app

client = TestClient(app)


def test_api_workflow():
    # 1. Health / Dashboard HTML
    index_res = client.get("/")
    assert index_res.status_code == 200

    # 2. Execute new run
    exec_res = client.post(
        "/api/runs/execute",
        json={"query": "Analyze GOOGL risk", "mode": "RECORD"},
    )
    assert exec_res.status_code == 200
    run_data = exec_res.json()
    run_id = run_data["run_id"]
    assert run_id

    # 3. List runs
    list_res = client.get("/api/runs")
    assert list_res.status_code == 200
    runs = list_res.json()
    assert any(r["run_id"] == run_id for r in runs)

    # 4. Get run details & cassette
    manifest_res = client.get(f"/api/runs/{run_id}")
    assert manifest_res.status_code == 200
    assert manifest_res.json()["run_id"] == run_id

    cassette_res = client.get(f"/api/runs/{run_id}/cassette")
    assert cassette_res.status_code == 200
    assert len(cassette_res.json()) >= 1

    snapshots_res = client.get(f"/api/runs/{run_id}/snapshots")
    assert snapshots_res.status_code == 200
    assert len(snapshots_res.json()) >= 1

    # 5. Deterministic replay
    replay_res = client.post(f"/api/runs/{run_id}/replay", json={"strict": True})
    assert replay_res.status_code == 200
    assert replay_res.json()["replayed"] is True

    # 6. Time-travel fork
    fork_res = client.post(
        f"/api/runs/{run_id}/fork",
        json={
            "run_id": run_id,
            "fork_at_step": 2,
            "patch_state": {"ticker": "GOOGL", "status": "circuit_breaker_triggered"},
        },
    )
    assert fork_res.status_code == 200
    fork_data = fork_res.json()
    fork_id = fork_data["run_id"]

    # 7. Diff
    diff_res = client.get(f"/api/diff?run_1_id={run_id}&run_2_id={fork_id}")
    assert diff_res.status_code == 200
    diff_data = diff_res.json()
    assert diff_data["run_1_id"] == run_id
    assert diff_data["run_2_id"] == fork_id
