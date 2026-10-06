"""Execution context and ContextVar tracking for virtualization and OTel propagation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional
import uuid

from agent_replay_harness.observability import generate_span_id, generate_trace_id
from agent_replay_harness.schemas import (
    AgentRunEvent,
    CassetteManifest,
    ExecutionMode,
    RunEventType,
)
from agent_replay_harness.storage.base import BaseTapeStorage


class ExecutionContext:
    """Active runtime context holding virtualization mode, tape storage, and step."""

    def __init__(
        self,
        run_id: str,
        mode: ExecutionMode,
        storage: BaseTapeStorage,
        parent_run_id: Optional[str] = None,
        fork_step: Optional[int] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        patch_rules: Optional[Dict[str, Any]] = None,
        strict_replay: bool = True,
    ) -> None:
        self.run_id = run_id
        self.mode = mode
        self.storage = storage
        self.parent_run_id = parent_run_id
        self.fork_step = fork_step
        self.trace_id = trace_id or generate_trace_id()
        self.span_id = span_id or generate_span_id()
        self.patch_rules = patch_rules or {}
        self.strict_replay = strict_replay

        self.step: int = fork_step if fork_step is not None else 0
        self.tool_call_counts: Dict[str, int] = {}
        self.total_tool_calls: int = 0

    def next_call_index(self, hash_key: str) -> int:
        idx = self.tool_call_counts.get(hash_key, 0)
        self.tool_call_counts[hash_key] = idx + 1
        self.total_tool_calls += 1
        return idx

    def advance_step(self, step_num: Optional[int] = None) -> int:
        if step_num is not None:
            self.step = step_num
        else:
            self.step += 1
        return self.step


_execution_context_var: ContextVar[Optional[ExecutionContext]] = ContextVar(
    "execution_context", default=None
)


def get_current_context() -> Optional[ExecutionContext]:
    """Retrieve the active ExecutionContext if within an execution_scope."""
    return _execution_context_var.get()


def set_current_context(ctx: ExecutionContext) -> Token:
    return _execution_context_var.set(ctx)


def reset_current_context(token: Token) -> None:
    _execution_context_var.reset(token)


@contextmanager
def execution_scope(
    run_id: Optional[str] = None,
    mode: ExecutionMode = ExecutionMode.RECORD,
    storage: Optional[BaseTapeStorage] = None,
    parent_run_id: Optional[str] = None,
    fork_step: Optional[int] = None,
    trace_id: Optional[str] = None,
    patch_rules: Optional[Dict[str, Any]] = None,
    strict_replay: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Generator[ExecutionContext, None, None]:
    """Context manager binding virtualization mode, trace IDs, and event persistence."""
    from agent_replay_harness.storage.file import FileTapeStorage

    actual_run_id = run_id or str(uuid.uuid4())
    actual_storage = storage or FileTapeStorage()

    ctx = ExecutionContext(
        run_id=actual_run_id,
        mode=mode,
        storage=actual_storage,
        parent_run_id=parent_run_id,
        fork_step=fork_step,
        trace_id=trace_id,
        patch_rules=patch_rules,
        strict_replay=strict_replay,
    )

    manifest = CassetteManifest(
        run_id=actual_run_id,
        parent_run_id=parent_run_id,
        fork_step=fork_step,
        mode=mode,
        status="running",
        metadata=metadata or {},
    )
    actual_storage.create_run(manifest)

    # Record RUN_STARTED event
    actual_storage.append_event(
        AgentRunEvent(
            run_id=actual_run_id,
            parent_run_id=parent_run_id,
            fork_step=fork_step,
            step=ctx.step,
            event_type=RunEventType.RUN_STARTED,
            payload={
                "mode": mode.value,
                "parent_run_id": parent_run_id,
                "fork_step": fork_step,
                "patch_rules": patch_rules,
            },
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
        )
    )

    token = set_current_context(ctx)
    try:
        yield ctx
        # Success completion
        manifest.status = "completed"
        manifest.completed_at = datetime.now(timezone.utc)
        manifest.total_steps = ctx.step
        manifest.total_tool_calls = ctx.total_tool_calls
        actual_storage.update_run_manifest(manifest)

        actual_storage.append_event(
            AgentRunEvent(
                run_id=actual_run_id,
                parent_run_id=parent_run_id,
                fork_step=fork_step,
                step=ctx.step,
                event_type=RunEventType.RUN_COMPLETED,
                payload={"total_steps": ctx.step, "total_tool_calls": ctx.total_tool_calls},
                trace_id=ctx.trace_id,
                span_id=ctx.span_id,
            )
        )
    except Exception as exc:
        manifest.status = "failed"
        manifest.completed_at = datetime.now(timezone.utc)
        manifest.total_steps = ctx.step
        manifest.total_tool_calls = ctx.total_tool_calls
        manifest.error = str(exc)
        actual_storage.update_run_manifest(manifest)

        actual_storage.append_event(
            AgentRunEvent(
                run_id=actual_run_id,
                parent_run_id=parent_run_id,
                fork_step=fork_step,
                step=ctx.step,
                event_type=RunEventType.RUN_FAILED,
                payload={"error": str(exc), "error_type": type(exc).__name__},
                trace_id=ctx.trace_id,
                span_id=ctx.span_id,
            )
        )
        raise
    finally:
        reset_current_context(token)
