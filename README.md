# Deterministic Replay & Time-Travel Debugging Harness for Agent Workflows

A production-grade execution virtualization, VCR cassette interception, and time-travel debugging engine designed for **LangGraph** state machines and autonomous AI agent workflows.

---

## 🏛️ Architecture Blueprint

```
                      +---------------------------------------+
                      |       Developer CLI / Test Suite      |
                      +-------------------+-------------------+
                                          | (run / replay / fork)
                                          v
+------------------+         +------------+------------+         +------------------+
|                  |         |                         |         |                  |
|  Agent Loop      | <=====> | Execution Interceptor   | <=====> | External Tools   |
|  (LangGraph /    |         | (VCR Middleware Engine) |         | (APIs, DBs, Web) |
|   State Machine) |         |                         |         |                  |
+------------------+         +------------+------------+         +------------------+
                                          |
                                          | State & Tape Events
                                          v
                             +------------+------------+
                             | PostgreSQL / Redis      |
                             | Event Log & Snapshots   |
                             +-------------------------+
```

---

## 🚀 Key Capabilities

1. **State & Event Serialization Engine**
   - Captures every LangGraph node transition, tool call payload, and LLM output into an append-only JSON event store.
   - Snapshots the complete memory footprint and scratchpad state at step $k$.
   - Computes state deltas (`diff_from_parent`) between checkpoints.

2. **Network & Tool Virtualization (VCR Layer)**
   - **`RECORD` Mode**: Intercepts outgoing tool calls, hashes canonical representations of `(tool_name, args, kwargs)`, and stores request/response cassettes.
   - **`REPLAY` Mode**: Deterministically re-executes workflows by serving recorded cassettes, eliminating external API costs and network jitter.
   - **`PASSTHROUGH` Mode**: Live unvirtualized execution for direct testing.

3. **State Forking & Time-Travel Debugging (`FORK` Mode)**
   - Branch execution directly from snapshot $k$.
   - Alter system prompt, tool output, or user input with state/tool patches.
   - Resume execution forward to verify error recovery without re-running expensive steps $1 \dots k-1$.

4. **LangGraph Checkpoint Adapter (`TapeCheckpointSaver`)**
   - Seamless drop-in `BaseCheckpointSaver` integration.
   - Bridges LangGraph channels and versions to immutable event logs.
   - Supports `copy_thread` and time-travel thread branching.

5. **OpenTelemetry Observability**
   - Propagates `trace_id` and `span_id` across node boundaries and intercepted tools.

6. **Interactive Developer CLI & Web UI**
   - Full-featured CLI (`agent-harness run`, `replay`, `fork`, `inspect`, `diff`, `list`, `serve`).
   - Web Dashboard with step scrubber, state diff viewer, and 1-click fork modal.

---

## 📦 Installation

```bash
# Using uv or pip
uv pip install -e .
```

---

## ⚡ Quickstart

### 1. Run live agent workflow and record event tape
```bash
agent-harness run --query "Analyze TSLA risk and filings"
```

### 2. Replay deterministically with 0 API cost
```bash
agent-harness replay --run-id <RUN-UUID>
```

### 3. Time-travel fork & fault injection
Branch from step 2 and inject a market circuit breaker or tool error:
```bash
agent-harness replay --run-id <RUN-UUID> --fork-at-step 2 --patch '{"market_data": {"status": "circuit_breaker_triggered", "error": "Flash crash detected"}}'
```

### 4. Side-by-side trajectory diffing
```bash
agent-harness diff --run-1 <ORIGINAL-UUID> --run-2 <FORK-UUID>
```

### 5. Launch Interactive Web Dashboard
```bash
agent-harness serve --port 8000
```
Open `http://localhost:8000` to scrub through steps and branch execution interactively.
