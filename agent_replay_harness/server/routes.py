"""FastAPI REST API routes for runs, replays, forking, and diffs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent_replay_harness.runner import fork_agent, replay_agent, run_agent
from agent_replay_harness.schemas import (
    AgentRunEvent,
    CassetteManifest,
    DiffResult,
    ExecutionMode,
    ForkRequest,
    RunSummary,
    StateSnapshot,
    ToolCallRecord,
)
from agent_replay_harness.storage.file import FileTapeStorage

router = APIRouter(prefix="/api")
storage = FileTapeStorage()


class ExecuteRequest(BaseModel):
    query: str = "Analyze TSLA risk and filings"
    mode: ExecutionMode = ExecutionMode.RECORD


class ReplayRequest(BaseModel):
    strict: bool = True


@router.get("/runs", response_model=List[RunSummary])
def list_runs() -> List[RunSummary]:
    """Retrieve all recorded runs."""
    return storage.list_runs()


@router.get("/runs/{run_id}", response_model=CassetteManifest)
def get_run(run_id: str) -> CassetteManifest:
    """Retrieve run manifest metadata."""
    manifest = storage.get_run_manifest(run_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Run not found")
    return manifest


@router.get("/runs/{run_id}/events", response_model=List[AgentRunEvent])
def get_run_events(run_id: str) -> List[AgentRunEvent]:
    """Retrieve append-only event log for a run."""
    return storage.get_events(run_id)


@router.get("/runs/{run_id}/snapshots", response_model=List[StateSnapshot])
def get_run_snapshots(run_id: str) -> List[StateSnapshot]:
    """Retrieve complete step-by-step state snapshots."""
    return storage.list_snapshots(run_id)


@router.get("/runs/{run_id}/cassette", response_model=List[ToolCallRecord])
def get_run_cassette(run_id: str) -> List[ToolCallRecord]:
    """Retrieve all recorded tool calls in cassette tape."""
    return storage.list_tool_calls(run_id)


@router.post("/runs/execute")
def execute_run(req: ExecuteRequest) -> Dict[str, Any]:
    """Execute new run in RECORD or PASSTHROUGH mode."""
    return run_agent(query=req.query, mode=req.mode, storage=storage)


@router.post("/runs/{run_id}/replay")
def replay_run(run_id: str, req: ReplayRequest) -> Dict[str, Any]:
    """Deterministically replay a run from tape."""
    try:
        return replay_agent(run_id=run_id, storage=storage, strict=req.strict)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/runs/{run_id}/fork")
def fork_run(run_id: str, req: ForkRequest) -> Dict[str, Any]:
    """Time-travel fork execution at step k with state/tool patches."""
    try:
        return fork_agent(
            parent_run_id=run_id,
            fork_at_step=req.fork_at_step,
            patch_state=req.patch_state,
            patch_tool_results=req.patch_tool_results,
            new_run_id=req.new_run_id,
            storage=storage,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/diff", response_model=DiffResult)
def diff_runs(run_1_id: str = Query(...), run_2_id: str = Query(...)) -> DiffResult:
    """Compare two runs side-by-side."""
    return storage.diff_runs(run_1_id, run_2_id)
