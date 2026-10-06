"""Unit tests for the LangGraph TapeCheckpointSaver adapter."""

from typing import TypedDict
from langgraph.graph import END, START, StateGraph

from agent_replay_harness.checkpointer import TapeCheckpointSaver


class SampleState(TypedDict):
    count: int
    tags: list


def step_increment(state: SampleState):
    return {"count": state["count"] + 1, "tags": state["tags"] + ["inc"]}


def step_double(state: SampleState):
    return {"count": state["count"] * 2, "tags": state["tags"] + ["double"]}


def test_tape_checkpointer_saves_snapshots(file_storage):
    checkpointer = TapeCheckpointSaver(storage=file_storage)
    builder = StateGraph(SampleState)
    builder.add_node("inc", step_increment)
    builder.add_node("double", step_double)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", "double")
    builder.add_edge("double", END)
    graph = builder.compile(checkpointer=checkpointer)

    run_id = "test-graph-1"
    config = {"configurable": {"thread_id": run_id, "checkpoint_ns": ""}}
    result = graph.invoke({"count": 5, "tags": []}, config=config)

    assert result["count"] == 12  # (5 + 1) * 2 = 12
    assert result["tags"] == ["inc", "double"]

    # Verify snapshots persisted
    snaps = file_storage.list_snapshots(run_id)
    assert len(snaps) >= 3  # input, inc, double

    # Check state data preserved in snapshot
    final_snap = snaps[-1]
    assert final_snap.state_data["count"] == 12
    assert final_snap.state_data["tags"] == ["inc", "double"]


def test_tape_checkpointer_fork_and_patch(file_storage):
    checkpointer = TapeCheckpointSaver(storage=file_storage)
    builder = StateGraph(SampleState)
    builder.add_node("inc", step_increment)
    builder.add_node("double", step_double)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", "double")
    builder.add_edge("double", END)
    graph = builder.compile(checkpointer=checkpointer)

    parent_id = "orig-run"
    config = {"configurable": {"thread_id": parent_id, "checkpoint_ns": ""}}
    graph.invoke({"count": 1, "tags": []}, config=config)

    # Checkpoints: 0=input, 1=before inc, 2=after inc
    # Fork after inc (checkpoint index 2)
    forked_id, fork_cfg = checkpointer.fork_checkpoint(
        parent_run_id=parent_id,
        fork_step=2,
        patch_state={"count": 50},
    )

    # Resume graph execution on forked thread
    fork_result = graph.invoke(None, config=fork_cfg)
    # step 'double' executes on patched count 50 -> 100!
    assert fork_result["count"] == 100
