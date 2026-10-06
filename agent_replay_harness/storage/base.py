"""Abstract base storage interface for event tapes, snapshots, and tool cassettes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent_replay_harness.schemas import (
    AgentRunEvent,
    CassetteManifest,
    CassetteTape,
    DiffResult,
    RunSummary,
    StateSnapshot,
    ToolCallRecord,
)


class BaseTapeStorage(ABC):
    """Abstract interface defining operations for agent tape persistence."""

    @abstractmethod
    def create_run(self, manifest: CassetteManifest) -> None:
        """Initialize or register a new run manifest."""
        pass

    @abstractmethod
    def update_run_manifest(self, manifest: CassetteManifest) -> None:
        """Update run manifest status, completed_at, total counts, etc."""
        pass

    @abstractmethod
    def get_run_manifest(self, run_id: str) -> Optional[CassetteManifest]:
        """Fetch run manifest by run_id."""
        pass

    @abstractmethod
    def append_event(self, event: AgentRunEvent) -> None:
        """Append an agent trajectory event to the run's event log."""
        pass

    @abstractmethod
    def get_events(self, run_id: str) -> List[AgentRunEvent]:
        """Retrieve all events for a given run in chronological order."""
        pass

    @abstractmethod
    def save_snapshot(self, snapshot: StateSnapshot) -> None:
        """Persist a memory footprint snapshot at step k."""
        pass

    @abstractmethod
    def get_snapshot(self, run_id: str, step: int) -> Optional[StateSnapshot]:
        """Retrieve snapshot for a given run and step."""
        pass

    @abstractmethod
    def list_snapshots(self, run_id: str) -> List[StateSnapshot]:
        """List all snapshots for a run in step order."""
        pass

    @abstractmethod
    def record_tool_call(self, record: ToolCallRecord) -> None:
        """Save an intercepted tool call record into the cassette."""
        pass

    @abstractmethod
    def get_tool_call(
        self,
        run_id: str,
        hash_key: str,
        call_index: int = 0,
    ) -> Optional[ToolCallRecord]:
        """Look up recorded tool call matching hash_key and sequential index."""
        pass

    @abstractmethod
    def list_tool_calls(self, run_id: str) -> List[ToolCallRecord]:
        """List all recorded tool calls for a run."""
        pass

    @abstractmethod
    def get_tape(self, run_id: str) -> Optional[CassetteTape]:
        """Load the complete CassetteTape bundle for a run."""
        pass

    @abstractmethod
    def list_runs(self) -> List[RunSummary]:
        """List summaries of all recorded runs."""
        pass

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """Delete all tape data for a run."""
        pass

    def diff_runs(self, run_1_id: str, run_2_id: str) -> DiffResult:
        """Compare two runs, identifying diverging step, state diff, and tool divergence."""
        from agent_replay_harness.hashing import compute_dict_diff

        snaps_1 = {s.step: s for s in self.list_snapshots(run_1_id)}
        snaps_2 = {s.step: s for s in self.list_snapshots(run_2_id)}
        tools_1 = self.list_tool_calls(run_1_id)
        tools_2 = self.list_tool_calls(run_2_id)

        all_steps = sorted(set(snaps_1.keys()) | set(snaps_2.keys()))
        step_comparisons: List[Dict[str, Any]] = []
        diverged_at_step: Optional[int] = None

        for step in all_steps:
            s1 = snaps_1.get(step)
            s2 = snaps_2.get(step)
            diff = compute_dict_diff(
                s1.state_data if s1 else {},
                s2.state_data if s2 else {},
            )
            step_match = s1 is not None and s2 is not None and not diff
            if not step_match and diverged_at_step is None:
                diverged_at_step = step

            step_comparisons.append({
                "step": step,
                "node_1": s1.node_name if s1 else None,
                "node_2": s2.node_name if s2 else None,
                "match": step_match,
                "diff": diff,
            })

        max_tools = max(len(tools_1), len(tools_2))
        tool_comparisons: List[Dict[str, Any]] = []
        for i in range(max_tools):
            t1 = tools_1[i] if i < len(tools_1) else None
            t2 = tools_2[i] if i < len(tools_2) else None
            t_match = (
                t1 is not None
                and t2 is not None
                and t1.tool_name == t2.tool_name
                and t1.hash_key == t2.hash_key
                and t1.result == t2.result
            )
            tool_comparisons.append({
                "index": i,
                "tool_1": t1.tool_name if t1 else None,
                "tool_2": t2.tool_name if t2 else None,
                "hash_1": t1.hash_key if t1 else None,
                "hash_2": t2.hash_key if t2 else None,
                "match": t_match,
                "result_1": t1.result if t1 else None,
                "result_2": t2.result if t2 else None,
            })

        return DiffResult(
            run_1_id=run_1_id,
            run_2_id=run_2_id,
            diverged_at_step=diverged_at_step,
            step_comparisons=step_comparisons,
            tool_call_comparisons=tool_comparisons,
        )
