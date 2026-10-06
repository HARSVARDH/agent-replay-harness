"""LangGraph checkpointer adapter connecting StateGraph state to the tape storage engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.memory import MemorySaver

from agent_replay_harness.context import get_current_context
from agent_replay_harness.hashing import compute_dict_diff
from agent_replay_harness.schemas import (
    AgentRunEvent,
    ExecutionMode,
    RunEventType,
    StateSnapshot,
)
from agent_replay_harness.storage.base import BaseTapeStorage
from agent_replay_harness.storage.file import FileTapeStorage


class TapeCheckpointSaver(MemorySaver):
    """Custom LangGraph checkpointer recording state snapshots to an append-only event tape."""

    def __init__(
        self,
        storage: Optional[BaseTapeStorage] = None,
    ) -> None:
        super().__init__()
        self.tape_storage = storage or FileTapeStorage()
        self._thread_step_counters: Dict[str, int] = {}
        self._last_snapshot_by_thread: Dict[str, StateSnapshot] = {}

    def copy_thread(self, target_thread_id: str, source_thread_id: str) -> None:
        """Copy checkpoints and channel blobs from source thread to target thread."""
        # Copy blobs
        for (tid, tns, k, v), val in list(self.blobs.items()):
            if tid == source_thread_id:
                self.blobs[(target_thread_id, tns, k, v)] = val

        # Copy storage
        if source_thread_id in self.storage:
            for ns, cps in self.storage[source_thread_id].items():
                self.storage[target_thread_id][ns] = dict(cps)

        # Copy writes
        for (tid, tns, cpid), write_list in list(self.writes.items()):
            if tid == source_thread_id:
                self.writes[(target_thread_id, tns, cpid)] = dict(write_list)

        # Copy step counter
        if source_thread_id in self._thread_step_counters:
            self._thread_step_counters[target_thread_id] = self._thread_step_counters[source_thread_id]
        if source_thread_id in self._last_snapshot_by_thread:
            self._last_snapshot_by_thread[target_thread_id] = self._last_snapshot_by_thread[source_thread_id]

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist checkpoint and record state snapshot and events to tape storage."""
        # Call base MemorySaver to maintain internal LangGraph state
        updated_config = super().put(config, checkpoint, metadata, new_versions)

        thread_id: str = config["configurable"]["thread_id"]
        ctx = get_current_context()

        # Determine step number
        current_step = self._thread_step_counters.get(thread_id, 0) + 1
        self._thread_step_counters[thread_id] = current_step
        if ctx:
            ctx.advance_step(current_step)

        # Extract node name and clean channel values
        node_name = metadata.get("source", "node")
        if node_name == "loop" and "langgraph_node" in metadata:
            node_name = metadata["langgraph_node"]

        # Channel values representing memory & scratchpad
        channel_values = dict(checkpoint.get("channel_values", {}))
        # Filter internal LangGraph branch keys for a cleaner scratchpad snapshot
        clean_state = {
            k: v for k, v in channel_values.items()
            if not k.startswith("branch:to:") and not k.startswith("__")
        }

        # Compute diff from previous snapshot
        prev_snapshot = self._last_snapshot_by_thread.get(thread_id)
        diff_from_parent = (
            compute_dict_diff(prev_snapshot.state_data, clean_state)
            if prev_snapshot
            else None
        )

        snapshot = StateSnapshot(
            snapshot_id=checkpoint["id"],
            run_id=thread_id,
            step=current_step,
            node_name=node_name,
            state_data=clean_state,
            parent_snapshot_id=prev_snapshot.snapshot_id if prev_snapshot else None,
            diff_from_parent=diff_from_parent,
        )
        self._last_snapshot_by_thread[thread_id] = snapshot

        # Persist snapshot to tape storage
        self.tape_storage.save_snapshot(snapshot)

        # Append STATE_SNAPSHOT event
        trace_id = ctx.trace_id if ctx else None
        span_id = ctx.span_id if ctx else None
        parent_run_id = ctx.parent_run_id if ctx else None
        fork_step = ctx.fork_step if ctx else None

        self.tape_storage.append_event(
            AgentRunEvent(
                run_id=thread_id,
                parent_run_id=parent_run_id,
                fork_step=fork_step,
                step=current_step,
                event_type=RunEventType.STATE_SNAPSHOT,
                payload={
                    "node_name": node_name,
                    "snapshot_id": snapshot.snapshot_id,
                    "state_summary": {k: type(v).__name__ for k, v in clean_state.items()},
                    "diff": diff_from_parent,
                },
                trace_id=trace_id,
                span_id=span_id,
            )
        )

        return updated_config

    def get_checkpoints_for_thread(self, thread_id: str) -> List[CheckpointTuple]:
        """Return all checkpoints for a thread ordered chronologically."""
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }
        # list() returns in reverse-chronological order, so reverse it
        cps = list(self.list(config))
        cps.reverse()
        return cps

    def fork_checkpoint(
        self,
        parent_run_id: str,
        fork_step: int,
        patch_state: Optional[Dict[str, Any]] = None,
        new_run_id: Optional[str] = None,
    ) -> Tuple[str, RunnableConfig]:
        """Time-travel fork: clone state at step k, apply patch, and return new config."""
        forked_run_id = new_run_id or f"{parent_run_id[:8]}-fork-{uuid.uuid4().hex[:6]}"
        cps = self.get_checkpoints_for_thread(parent_run_id)

        if not cps:
            # Check if snapshots exist on disk to hydrate
            snapshots = self.tape_storage.list_snapshots(parent_run_id)
            if not snapshots:
                raise ValueError(f"No checkpoints or snapshots found for parent run '{parent_run_id}'.")

        self.copy_thread(forked_run_id, parent_run_id)
        self._thread_step_counters[forked_run_id] = fork_step

        # Find target checkpoint at step
        target_cp: Optional[CheckpointTuple] = None
        if fork_step < len(cps):
            target_cp = cps[fork_step]
        elif cps:
            target_cp = cps[-1]

        if target_cp is None:
            raise ValueError(f"Checkpoint for step {fork_step} could not be resolved.")

        # Copy snapshots up to fork_step to maintain historical trajectory in forked run
        parent_snaps = self.tape_storage.list_snapshots(parent_run_id)
        last_inherited_snap = None
        seen_steps = set()
        for s in parent_snaps:
            if s.step < fork_step and s.step not in seen_steps:
                seen_steps.add(s.step)
                inherited = StateSnapshot(
                    snapshot_id=s.snapshot_id,
                    run_id=forked_run_id,
                    step=s.step,
                    node_name=s.node_name,
                    state_data=s.state_data,
                    parent_snapshot_id=s.parent_snapshot_id,
                    diff_from_parent=s.diff_from_parent,
                )
                self.tape_storage.save_snapshot(inherited)
                last_inherited_snap = inherited

        if last_inherited_snap:
            self._last_snapshot_by_thread[forked_run_id] = last_inherited_snap

        # Emit FORK_CREATED event
        self.tape_storage.append_event(
            AgentRunEvent(
                run_id=forked_run_id,
                parent_run_id=parent_run_id,
                fork_step=fork_step,
                step=fork_step,
                event_type=RunEventType.FORK_CREATED,
                payload={
                    "parent_run_id": parent_run_id,
                    "fork_step": fork_step,
                    "patch_state": patch_state,
                },
            )
        )

        fork_config: RunnableConfig = {
            "configurable": {
                "thread_id": forked_run_id,
                "checkpoint_ns": "",
                "checkpoint_id": target_cp.checkpoint["id"],
            }
        }

        # Apply patch if provided
        if patch_state:
            # Modify the channel values of the checkpoint in storage
            cp_id = target_cp.checkpoint["id"]
            ns = ""
            if forked_run_id in self.storage and ns in self.storage[forked_run_id]:
                if cp_id in self.storage[forked_run_id][ns]:
                    raw_cp, raw_meta, parent_id = self.storage[forked_run_id][ns][cp_id]
                    cp_dict: Checkpoint = self.serde.loads_typed(raw_cp)
                    cp_dict["channel_values"] = self._load_blobs(
                        forked_run_id, ns, cp_dict["channel_versions"]
                    )
                    cp_dict["channel_values"].update(patch_state)

                    # Update blobs for patched keys
                    for k, v in patch_state.items():
                        current_ver = cp_dict["channel_versions"].get(k)
                        ver = self.get_next_version(current_ver, channel=k)
                        cp_dict["channel_versions"][k] = ver
                        self.blobs[(forked_run_id, ns, k, ver)] = self.serde.dumps_typed(v)

                    # Save updated checkpoint back to storage
                    self.storage[forked_run_id][ns][cp_id] = (
                        self.serde.dumps_typed(cp_dict),
                        raw_meta,
                        parent_id,
                    )

                    # Record fork snapshot at fork_step
                    clean_vals = {
                        k: v for k, v in cp_dict["channel_values"].items()
                        if not k.startswith("branch:to:") and not k.startswith("__")
                    }
                    fork_snap = StateSnapshot(
                        snapshot_id=f"fork-{cp_id}",
                        run_id=forked_run_id,
                        step=fork_step,
                        node_name="fork_patch",
                        state_data=clean_vals,
                        parent_snapshot_id=parent_id,
                        diff_from_parent={"patch": patch_state},
                    )
                    self.tape_storage.save_snapshot(fork_snap)
                    self._last_snapshot_by_thread[forked_run_id] = fork_snap

        return forked_run_id, fork_config
