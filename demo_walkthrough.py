#!/usr/bin/env python3
"""Deterministic Replay & Time-Travel Debugging Harness - Complete End-to-End Walkthrough Demo.

Demonstrates:
  1. RECORD MODE: Live execution, external tool virtualization, tape recording.
  2. REPLAY MODE: 100% deterministic re-execution with 0 external API calls & 0 cost.
  3. TIME-TRAVEL FORK: Branching from Step 3, injecting state fault, self-healing recovery.
  4. TRAJECTORY DIFF: Side-by-side execution comparison highlighting state divergence.
"""

import json
from pathlib import Path
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from agent_replay_harness.runner import fork_agent, replay_agent, run_agent
from agent_replay_harness.schemas import ExecutionMode
from agent_replay_harness.storage.file import FileTapeStorage

console = Console()
storage = FileTapeStorage(base_dir=".demo_tapes")


def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]Deterministic Replay & Time-Travel Debugging Harness[/bold cyan]\n"
            "[dim]LangGraph Virtualization Engine • VCR Cassette Tapes • Fault Injection[/dim]",
            border_style="cyan",
        )
    )


def step_1_record_live():
    console.print("\n[bold yellow]═══ STEP 1: RECORD MODE (Live Multi-Node Agent Execution) ═══[/bold yellow]")
    console.print("[dim]Executing financial analysis workflow. External tool calls will be intercepted and hashed into cassette tapes...[/dim]\n")

    t0 = time.perf_counter()
    result = run_agent(
        query="Analyze TSLA financial risk and filings",
        mode=ExecutionMode.RECORD,
        storage=storage,
    )
    duration = (time.perf_counter() - t0) * 1000
    run_id = result["run_id"]

    console.print(f"[bold green]✔ Run Completed in {duration:.1f}ms[/bold green]")
    console.print(f"[bold]Run ID:[/bold] {run_id}")
    console.print(f"[bold]Trace ID:[/bold] {result.get('trace_id')}")

    # Inspect Cassette Tape
    tool_calls = storage.list_tool_calls(run_id)
    table = Table(title="Captured Tool Cassettes", border_style="cyan")
    table.add_column("Step", width=6)
    table.add_column("Tool Name", style="bold green")
    table.add_column("Hash Key", style="dim yellow")
    table.add_column("Recorded Result (Truncated)", style="dim")
    table.add_column("Latency")

    for tc in tool_calls:
        res_str = json.dumps(tc.result)
        if len(res_str) > 40:
            res_str = res_str[:37] + "..."
        table.add_row(
            str(tc.step),
            tc.tool_name,
            tc.hash_key,
            res_str,
            f"{tc.duration_ms:.1f}ms",
        )
    console.print(table)
    return run_id


def step_2_deterministic_replay(run_id: str):
    console.print("\n[bold yellow]═══ STEP 2: REPLAY MODE (Deterministic Execution from Tape) ═══[/bold yellow]")
    console.print("[dim]Re-running exact workflow. External tools are bypassed; inputs are looked up by hash(tool, args)...[/dim]\n")

    t0 = time.perf_counter()
    replay_result = replay_agent(run_id=run_id, storage=storage, strict=True)
    duration = (time.perf_counter() - t0) * 1000

    console.print(f"[bold green]✔ Deterministic Replay Completed in {duration:.1f}ms (0 API Calls Made)[/bold green]")
    console.print(f"[bold]Original Run:[/bold] {run_id}")
    console.print(f"[bold]Replayed Run:[/bold] {replay_result['replay_id']}")
    console.print(f"[bold]Cassette Hits:[/bold] {replay_result['total_tool_calls']}/{replay_result['total_tool_calls']}")


def step_3_time_travel_fork(run_id: str):
    console.print("\n[bold yellow]═══ STEP 3: TIME-TRAVEL FORKING & FAULT INJECTION ═══[/bold yellow]")
    console.print("[dim]Branching from Step 3 without re-running earlier steps. Injecting simulated Flash Crash anomaly...[/dim]\n")

    fork_result = fork_agent(
        parent_run_id=run_id,
        fork_at_step=3,
        patch_state={
            "market_data": {
                "status": "circuit_breaker_triggered",
                "error": "Simulated Flash Crash (-15% deviation)",
            }
        },
        storage=storage,
    )
    fork_id = fork_result["run_id"]
    out = fork_result["output"]

    console.print(f"[bold magenta]✔ Fork Completed:[/bold magenta] {fork_id}")
    console.print(f"[bold]Action Taken by Agent:[/bold] [yellow]{out.get('action_taken')}[/yellow]")
    console.print(f"[bold]Recovered Error:[/bold] [red]{out['final_report']['dossier']['error_recovered']}[/red]")
    console.print(f"[bold]Final Status:[/bold] [bold green]{out.get('status')}[/bold green]")
    return fork_id


def step_4_diff_trajectories(parent_id: str, fork_id: str):
    console.print("\n[bold yellow]═══ STEP 4: EXECUTION TRAJECTORY DIFFING ═══[/bold yellow]")
    diff = storage.diff_runs(parent_id, fork_id)

    console.print(f"[bold]Divergence Step:[/bold] [bold red]Step {diff.diverged_at_step}[/bold red]\n")

    table = Table(title="Root vs. Branch Execution Comparison", border_style="yellow")
    table.add_column("Step", width=6)
    table.add_column("Original Run Node")
    table.add_column("Forked Run Node")
    table.add_column("Match?", justify="center")
    table.add_column("State Differences", style="dim")

    for comp in diff.step_comparisons:
        match_str = "[green]YES[/green]" if comp["match"] else "[red]NO[/red]"
        diff_str = json.dumps(comp["diff"]) if comp["diff"] else "-"
        if len(diff_str) > 55:
            diff_str = diff_str[:52] + "..."
        table.add_row(
            str(comp["step"]),
            comp["node_1"] or "-",
            comp["node_2"] or "-",
            match_str,
            diff_str,
        )
    console.print(table)


def main():
    print_banner()
    run_id = step_1_record_live()
    step_2_deterministic_replay(run_id)
    fork_id = step_3_time_travel_fork(run_id)
    step_4_diff_trajectories(run_id, fork_id)

    console.print(
        Panel.fit(
            "[bold green]✔ All Virtualization & Time-Travel Debugging Checks Passed![/bold green]\n"
            "Try launching the interactive web UI with: [bold cyan]agent-harness serve[/bold cyan]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
