"""High-level runner engine orchestrating Record, Replay, and Time-Travel Forking."""

from __future__ import annotations

from typing import Any, Dict, Optional
import uuid

from agent_replay_harness.agent.workflow import build_agent_graph
from agent_replay_harness.checkpointer import TapeCheckpointSaver
from agent_replay_harness.context import execution_scope
from agent_replay_harness.schemas import ExecutionMode
from agent_replay_harness.storage.base import BaseTapeStorage
from agent_replay_harness.storage.file import FileTapeStorage


def run_agent(
    query: str = "Analyze TSLA risk and filings",
    mode: ExecutionMode = ExecutionMode.RECORD,
    storage: Optional[BaseTapeStorage] = None,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the agent workflow in RECORD or PASSTHROUGH mode."""
    actual_storage = storage or FileTapeStorage()
    actual_run_id = run_id or str(uuid.uuid4())
    checkpointer = TapeCheckpointSaver(storage=actual_storage)
    graph = build_agent_graph(checkpointer=checkpointer)

    run_meta = {
        "query": query,
        **(metadata or {}),
    }

    with execution_scope(
        run_id=actual_run_id,
        mode=mode,
        storage=actual_storage,
        metadata=run_meta,
    ) as ctx:
        config = {
            "configurable": {
                "thread_id": actual_run_id,
                "checkpoint_ns": "",
            }
        }
        result = graph.invoke({"query": query}, config=config)
        return {
            "run_id": actual_run_id,
            "mode": mode.value,
            "output": result,
            "total_steps": ctx.step,
            "total_tool_calls": ctx.total_tool_calls,
            "trace_id": ctx.trace_id,
        }


def replay_agent(
    run_id: str,
    storage: Optional[BaseTapeStorage] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Deterministically replay a recorded historical run entirely from cassette tape."""
    actual_storage = storage or FileTapeStorage()
    manifest = actual_storage.get_run_manifest(run_id)
    if not manifest:
        raise ValueError(f"Run ID '{run_id}' not found in storage.")

    # Retrieve initial query from manifest or first event
    query = manifest.metadata.get("query", "Analyze TSLA risk and filings")
    replay_run_id = f"replay-{run_id[:8]}-{uuid.uuid4().hex[:4]}"

    # Use checkpointer initialized with storage
    checkpointer = TapeCheckpointSaver(storage=actual_storage)
    graph = build_agent_graph(checkpointer=checkpointer)

    with execution_scope(
        run_id=replay_run_id,
        mode=ExecutionMode.REPLAY,
        storage=actual_storage,
        parent_run_id=run_id,
        strict_replay=strict,
        metadata={"replaying_run_id": run_id, "query": query},
    ) as ctx:
        config = {
            "configurable": {
                "thread_id": replay_run_id,
                "checkpoint_ns": "",
            }
        }
        result = graph.invoke({"query": query}, config=config)
        return {
            "run_id": run_id,
            "replay_id": replay_run_id,
            "mode": ExecutionMode.REPLAY.value,
            "output": result,
            "total_steps": ctx.step,
            "total_tool_calls": ctx.total_tool_calls,
            "replayed": True,
        }


def fork_agent(
    parent_run_id: str,
    fork_at_step: int,
    patch_state: Optional[Dict[str, Any]] = None,
    patch_tool_results: Optional[Dict[str, Any]] = None,
    new_run_id: Optional[str] = None,
    storage: Optional[BaseTapeStorage] = None,
) -> Dict[str, Any]:
    """Time-travel execution branching: fork from step k, apply state/tool patches, and resume."""
    actual_storage = storage or FileTapeStorage()
    manifest = actual_storage.get_run_manifest(parent_run_id)
    if not manifest:
        raise ValueError(f"Parent run ID '{parent_run_id}' not found in storage.")

    checkpointer = TapeCheckpointSaver(storage=actual_storage)

    # First, run a baseline execution on the checkpointer to populate memory checkpoints
    # if this process is fresh
    if parent_run_id not in checkpointer.storage:
        # Hydrate parent state by running in replay mode
        query = manifest.metadata.get("query", "Analyze TSLA risk and filings")
        graph_init = build_agent_graph(checkpointer=checkpointer)
        with execution_scope(
            run_id=parent_run_id,
            mode=ExecutionMode.REPLAY,
            storage=actual_storage,
            strict_replay=False,
        ):
            graph_init.invoke(
                {"query": query},
                config={"configurable": {"thread_id": parent_run_id, "checkpoint_ns": ""}},
            )

    # Now fork checkpoint at step k
    forked_run_id, fork_config = checkpointer.fork_checkpoint(
        parent_run_id=parent_run_id,
        fork_step=fork_at_step,
        patch_state=patch_state,
        new_run_id=new_run_id,
    )

    graph = build_agent_graph(checkpointer=checkpointer)

    with execution_scope(
        run_id=forked_run_id,
        mode=ExecutionMode.FORK,
        storage=actual_storage,
        parent_run_id=parent_run_id,
        fork_step=fork_at_step,
        patch_rules=patch_tool_results,
        metadata={
            "parent_run_id": parent_run_id,
            "fork_at_step": fork_at_step,
            "patch_state": patch_state,
            "patch_tool_results": patch_tool_results,
        },
    ) as ctx:
        # Resume execution forward from the patched checkpoint
        result = graph.invoke(None, config=fork_config)
        return {
            "run_id": forked_run_id,
            "parent_run_id": parent_run_id,
            "fork_at_step": fork_at_step,
            "mode": ExecutionMode.FORK.value,
            "output": result,
            "total_steps": ctx.step,
            "total_tool_calls": ctx.total_tool_calls,
            "patched_state": patch_state,
            "patched_tools": patch_tool_results,
        }
