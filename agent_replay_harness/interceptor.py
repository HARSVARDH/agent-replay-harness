"""VCR interceptor engine and @record_tool_call decorator for deterministic replay."""

from __future__ import annotations

import asyncio
from functools import wraps
import inspect
import time
from typing import Any, Callable, Dict, Optional

from agent_replay_harness.context import get_current_context
from agent_replay_harness.hashing import hash_tool_call
from agent_replay_harness.observability import trace_span
from agent_replay_harness.schemas import (
    AgentRunEvent,
    ExecutionMode,
    RunEventType,
    ToolCallRecord,
)


class CassetteNotFoundError(Exception):
    """Raised when replaying a tool call with no matching entry on tape."""

    def __init__(self, tool_name: str, hash_key: str, call_index: int, run_id: str) -> None:
        super().__init__(
            f"VCR Cassette missing for tool '{tool_name}' with hash '{hash_key}' "
            f"(call_index={call_index}) in run '{run_id}'."
        )
        self.tool_name = tool_name
        self.hash_key = hash_key
        self.call_index = call_index
        self.run_id = run_id


class CassetteReplayError(Exception):
    """Raised when replaying an intercepted error recorded on tape."""
    pass


def record_tool_call(name: Optional[str] = None) -> Callable:
    """Decorator intercepting tool calls for RECORD, REPLAY, PASSTHROUGH, and FORK modes."""

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = get_current_context()
                if ctx is None or ctx.mode == ExecutionMode.PASSTHROUGH:
                    return await func(*args, **kwargs)

                hash_key = hash_tool_call(tool_name, args, kwargs)
                call_index = ctx.next_call_index(hash_key)

                # Check for active patch rules (Fault Injection / Forking)
                patched, patch_val, patch_err = _check_patch(ctx, tool_name, kwargs)
                if patched:
                    return _handle_patched_result(
                        ctx, tool_name, args, kwargs, hash_key, call_index, patch_val, patch_err
                    )

                if ctx.mode == ExecutionMode.REPLAY:
                    return _handle_replay(
                        ctx, tool_name, args, kwargs, hash_key, call_index
                    )

                if ctx.mode == ExecutionMode.FORK:
                    # If step is before or at fork_step, check parent cassette first
                    if ctx.parent_run_id and ctx.step <= (ctx.fork_step or 0):
                        parent_record = ctx.storage.get_tool_call(
                            ctx.parent_run_id, hash_key, call_index
                        )
                        if parent_record is not None:
                            return _serve_from_record(ctx, parent_record, tool_name, args, kwargs)

                # RECORD mode or live execution in FORK mode
                start_time = time.perf_counter()
                with trace_span(f"tool:{tool_name}", {"tool.name": tool_name, "hash_key": hash_key}) as span_info:
                    try:
                        result = await func(*args, **kwargs)
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        _record_success(
                            ctx, tool_name, args, kwargs, result, duration_ms, hash_key, call_index, span_info
                        )
                        return result
                    except Exception as exc:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        _record_failure(
                            ctx, tool_name, args, kwargs, exc, duration_ms, hash_key, call_index, span_info
                        )
                        raise

            return async_wrapper

        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = get_current_context()
                if ctx is None or ctx.mode == ExecutionMode.PASSTHROUGH:
                    return func(*args, **kwargs)

                hash_key = hash_tool_call(tool_name, args, kwargs)
                call_index = ctx.next_call_index(hash_key)

                # Check for active patch rules (Fault Injection / Forking)
                patched, patch_val, patch_err = _check_patch(ctx, tool_name, kwargs)
                if patched:
                    return _handle_patched_result(
                        ctx, tool_name, args, kwargs, hash_key, call_index, patch_val, patch_err
                    )

                if ctx.mode == ExecutionMode.REPLAY:
                    return _handle_replay(
                        ctx, tool_name, args, kwargs, hash_key, call_index
                    )

                if ctx.mode == ExecutionMode.FORK:
                    if ctx.parent_run_id and ctx.step <= (ctx.fork_step or 0):
                        parent_record = ctx.storage.get_tool_call(
                            ctx.parent_run_id, hash_key, call_index
                        )
                        if parent_record is not None:
                            return _serve_from_record(ctx, parent_record, tool_name, args, kwargs)

                # RECORD mode or live execution in FORK mode
                start_time = time.perf_counter()
                with trace_span(f"tool:{tool_name}", {"tool.name": tool_name, "hash_key": hash_key}) as span_info:
                    try:
                        result = func(*args, **kwargs)
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        _record_success(
                            ctx, tool_name, args, kwargs, result, duration_ms, hash_key, call_index, span_info
                        )
                        return result
                    except Exception as exc:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        _record_failure(
                            ctx, tool_name, args, kwargs, exc, duration_ms, hash_key, call_index, span_info
                        )
                        raise

            return sync_wrapper

    return decorator


def _check_patch(
    ctx: Any, tool_name: str, kwargs: Dict[str, Any]
) -> tuple[bool, Any, Optional[str]]:
    """Check if the context has an active patch for this tool or tool_result."""
    if not ctx.patch_rules:
        return False, None, None

    # Check for direct tool patch
    if tool_name in ctx.patch_rules:
        rule = ctx.patch_rules[tool_name]
        if isinstance(rule, dict) and "error" in rule:
            return True, None, rule["error"]
        return True, rule, None

    # Check for generic 'tool_result' patch
    if "tool_result" in ctx.patch_rules:
        val = ctx.patch_rules["tool_result"]
        return True, val, None

    # Check for generic 'tool_error' patch
    if "tool_error" in ctx.patch_rules:
        return True, None, str(ctx.patch_rules["tool_error"])

    return False, None, None


def _handle_patched_result(
    ctx: Any,
    tool_name: str,
    args: Any,
    kwargs: Any,
    hash_key: str,
    call_index: int,
    patch_val: Any,
    patch_err: Optional[str],
) -> Any:
    """Record and return injected/patched tool outcome."""
    record = ToolCallRecord(
        run_id=ctx.run_id,
        step=ctx.step,
        tool_name=tool_name,
        args=list(args),
        kwargs=kwargs,
        result=patch_val,
        error=patch_err,
        duration_ms=0.5,
        hash_key=hash_key,
        call_index=call_index,
        trace_id=ctx.trace_id,
        span_id=ctx.span_id,
        replayed=False,
    )
    ctx.storage.record_tool_call(record)

    ctx.storage.append_event(
        AgentRunEvent(
            run_id=ctx.run_id,
            parent_run_id=ctx.parent_run_id,
            fork_step=ctx.fork_step,
            step=ctx.step,
            event_type=RunEventType.TOOL_CALL,
            payload={
                "tool_name": tool_name,
                "patched": True,
                "result": patch_val,
                "error": patch_err,
                "hash_key": hash_key,
                "call_index": call_index,
            },
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
        )
    )

    if patch_err:
        raise RuntimeError(f"Patched fault injected: {patch_err}")
    return patch_val


def _handle_replay(
    ctx: Any,
    tool_name: str,
    args: Any,
    kwargs: Any,
    hash_key: str,
    call_index: int,
) -> Any:
    """Lookup and return recorded tool response from cassette tape."""
    record = ctx.storage.get_tool_call(ctx.run_id, hash_key, call_index)
    if record is None and ctx.parent_run_id:
        record = ctx.storage.get_tool_call(ctx.parent_run_id, hash_key, call_index)

    if record is None:
        if ctx.strict_replay:
            raise CassetteNotFoundError(tool_name, hash_key, call_index, ctx.run_id)
        return None

    return _serve_from_record(ctx, record, tool_name, args, kwargs)


def _serve_from_record(
    ctx: Any,
    record: ToolCallRecord,
    tool_name: str,
    args: Any,
    kwargs: Any,
) -> Any:
    """Emit replay event and return payload or re-raise error from record."""
    ctx.storage.append_event(
        AgentRunEvent(
            run_id=ctx.run_id,
            parent_run_id=ctx.parent_run_id,
            fork_step=ctx.fork_step,
            step=ctx.step,
            event_type=RunEventType.TOOL_CALL,
            payload={
                "tool_name": tool_name,
                "result": record.result,
                "error": record.error,
                "hash_key": record.hash_key,
                "call_index": record.call_index,
                "replayed": True,
                "original_duration_ms": record.duration_ms,
            },
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
        )
    )

    if record.error:
        raise CassetteReplayError(f"Replayed failure from tape: {record.error}")
    return record.result


def _record_success(
    ctx: Any,
    tool_name: str,
    args: Any,
    kwargs: Any,
    result: Any,
    duration_ms: float,
    hash_key: str,
    call_index: int,
    span_info: Dict[str, str],
) -> None:
    """Persist successful tool execution to storage."""
    record = ToolCallRecord(
        run_id=ctx.run_id,
        step=ctx.step,
        tool_name=tool_name,
        args=list(args),
        kwargs=kwargs,
        result=result,
        duration_ms=duration_ms,
        hash_key=hash_key,
        call_index=call_index,
        trace_id=span_info.get("trace_id", ctx.trace_id),
        span_id=span_info.get("span_id", ctx.span_id),
        replayed=False,
    )
    ctx.storage.record_tool_call(record)

    ctx.storage.append_event(
        AgentRunEvent(
            run_id=ctx.run_id,
            parent_run_id=ctx.parent_run_id,
            fork_step=ctx.fork_step,
            step=ctx.step,
            event_type=RunEventType.TOOL_CALL,
            payload={
                "tool_name": tool_name,
                "result": result,
                "duration_ms": duration_ms,
                "hash_key": hash_key,
                "call_index": call_index,
                "replayed": False,
            },
            trace_id=span_info.get("trace_id", ctx.trace_id),
            span_id=span_info.get("span_id", ctx.span_id),
        )
    )


def _record_failure(
    ctx: Any,
    tool_name: str,
    args: Any,
    kwargs: Any,
    exc: Exception,
    duration_ms: float,
    hash_key: str,
    call_index: int,
    span_info: Dict[str, str],
) -> None:
    """Persist failed tool execution to storage."""
    record = ToolCallRecord(
        run_id=ctx.run_id,
        step=ctx.step,
        tool_name=tool_name,
        args=list(args),
        kwargs=kwargs,
        error=str(exc),
        duration_ms=duration_ms,
        hash_key=hash_key,
        call_index=call_index,
        trace_id=span_info.get("trace_id", ctx.trace_id),
        span_id=span_info.get("span_id", ctx.span_id),
        replayed=False,
    )
    ctx.storage.record_tool_call(record)

    ctx.storage.append_event(
        AgentRunEvent(
            run_id=ctx.run_id,
            parent_run_id=ctx.parent_run_id,
            fork_step=ctx.fork_step,
            step=ctx.step,
            event_type=RunEventType.TOOL_CALL,
            payload={
                "tool_name": tool_name,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "duration_ms": duration_ms,
                "hash_key": hash_key,
                "call_index": call_index,
                "replayed": False,
            },
            trace_id=span_info.get("trace_id", ctx.trace_id),
            span_id=span_info.get("span_id", ctx.span_id),
        )
    )
