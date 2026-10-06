"""OpenTelemetry observability setup and trace context management."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Dict, Generator, List, Optional
import uuid

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode, format_span_id, format_trace_id

# Global in-memory exporter for inspecting captured traces locally
_in_memory_exporter = InMemorySpanExporter()

# Configure TracerProvider if not already configured
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_in_memory_exporter))
trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("agent-replay-harness", "0.1.0")


def generate_trace_id() -> str:
    """Generate a 32-character hex trace ID."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a 16-character hex span ID."""
    return uuid.uuid4().hex[:16]


def get_captured_spans() -> List[Any]:
    """Retrieve all in-memory captured OpenTelemetry spans."""
    return _in_memory_exporter.get_finished_spans()


def clear_captured_spans() -> None:
    """Clear captured in-memory OpenTelemetry spans."""
    _in_memory_exporter.clear()


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, str], None, None]:
    """Context manager creating an OpenTelemetry span and yielding its IDs."""
    attrs = attributes or {}
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, str(v))

        ctx = span.get_span_context()
        trace_id = format_trace_id(ctx.trace_id) if ctx.trace_id else generate_trace_id()
        span_id = format_span_id(ctx.span_id) if ctx.span_id else generate_span_id()

        try:
            yield {"trace_id": trace_id, "span_id": span_id}
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise
