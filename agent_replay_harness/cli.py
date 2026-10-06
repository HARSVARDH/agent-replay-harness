"""Developer CLI for Deterministic Replay, Time-Travel Debugging, and State Forking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
import typer

from agent_replay_harness.hashing import canonical_json_dumps
from agent_replay_harness.runner import fork_agent, replay_agent, run_agent
from agent_replay_harness.schemas import ExecutionMode
from agent_replay_harness.storage.file import FileTapeStorage
from agent_replay_harness.storage.sqlite import SQLiteTapeStorage

app = typer.Typer(
    name="agent-harness",
    help="Deterministic Replay & Time-Travel Debugging Harness for Agent Workflows",
    add_completion=False,
)
console = Console()


def _get_storage(backend: str = "file", storage_dir: str = ".tapes"):
    if backend == "sqlite":
        return SQLiteTapeStorage(Path(storage_dir) / "harness.db")
    return FileTapeStorage(storage_dir)


@app.command("run")
def run_command(
    query: str = typer.Option("Analyze TSLA risk and filings", "--query", "-q", help="Agent prompt or goal"),
    mode: str = typer.Option("RECORD", "--mode", "-m", help="RECORD or PASSTHROUGH"),
    backend: str = typer.Option("file", "--backend", "-b", help="Storage backend (file or sqlite)"),
    storage_dir: str = typer.Option(".tapes", "--storage-dir", "-d", help="Tapes directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Execute the agent workflow and capture the complete event tape and state snapshots."""
    storage = _get_storage(backend, storage_dir)
    exec_mode = ExecutionMode(mode.upper())

    with console.status(f"[bold cyan]Executing workflow in {exec_mode.value} mode...", spinner="dots"):
        result = run_agent(query=query, mode=exec_mode, storage=storage)

    if json_output:
        console.print_json(json.dumps(result, default=str))
        return

    run_id = result["run_id"]
    console.print(
        Panel.fit(
            f"[bold green]✔ Run Completed Successfully[/bold green]\n"
            f"[bold]Run ID:[/bold] {run_id}\n"
            f"[bold]Mode:[/bold] {result['mode']}\n"
            f"[bold]Total Steps:[/bold] {result['total_steps']} | [bold]Tool Calls:[/bold] {result['total_tool_calls']}\n"
            f"[bold]Trace ID:[/bold] {result.get('trace_id', 'N/A')}",
            title="[bold blue]Agent Replay Harness[/bold blue]",
            border_style="green",
        )
    )

    # Display steps tree
    snapshots = storage.list_snapshots(run_id)
    tree = Tree(f"[bold yellow]Execution Trajectory ({len(snapshots)} checkpoints)")
    for s in snapshots:
        tree.add(f"[bold cyan]Step {s.step}:[/bold cyan] [magenta]{s.node_name}[/magenta] (Status: {s.state_data.get('status', 'N/A')})")
    console.print(tree)

    # Display tool calls
    tool_calls = storage.list_tool_calls(run_id)
    if tool_calls:
        table = Table(title=f"Recorded Cassette Tapes ({len(tool_calls)} calls)", border_style="cyan")
        table.add_column("Step", style="dim", width=6)
        table.add_column("Tool Name", style="bold green")
        table.add_column("Hash Key", style="dim yellow")
        table.add_column("Latency", justify="right")
        table.add_column("Status", style="bold")
        for tc in tool_calls:
            status_text = "[red]ERROR[/red]" if tc.error else "[green]OK[/green]"
            table.add_row(
                str(tc.step),
                tc.tool_name,
                tc.hash_key,
                f"{tc.duration_ms:.1f}ms",
                status_text,
            )
        console.print(table)


@app.command("replay")
def replay_command(
    run_id: str = typer.Option(..., "--run-id", "-r", help="UUID of historical run to replay"),
    backend: str = typer.Option("file", "--backend", "-b", help="Storage backend (file or sqlite)"),
    storage_dir: str = typer.Option(".tapes", "--storage-dir", "-d", help="Tapes directory"),
    fork_at_step: Optional[int] = typer.Option(None, "--fork-at-step", help="Step number to branch execution from"),
    patch: Optional[str] = typer.Option(None, "--patch", help="JSON string to patch state or tool result"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Raise error if cassette missing"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Deterministically replay a recorded run with 0 API cost, or fork forward with patched state."""
    storage = _get_storage(backend, storage_dir)

    # If fork parameters are provided via replay command (per user checklist example)
    if fork_at_step is not None:
        patch_dict = json.loads(patch) if patch else {}
        # Differentiate state patch vs tool result patch
        patch_state = None
        patch_tools = None
        if "tool_result" in patch_dict or any("fetch" in k or "evaluate" in k for k in patch_dict):
            patch_tools = patch_dict
        else:
            patch_state = patch_dict

        with console.status(f"[bold magenta]Forking run {run_id} at step {fork_at_step}...", spinner="bouncingBar"):
            result = fork_agent(
                parent_run_id=run_id,
                fork_at_step=fork_at_step,
                patch_state=patch_state,
                patch_tool_results=patch_tools,
                storage=storage,
            )

        if json_output:
            console.print_json(json.dumps(result, default=str))
            return

        console.print(
            Panel.fit(
                f"[bold magenta]✔ Time-Travel Fork Completed[/bold magenta]\n"
                f"[bold]New Fork Run ID:[/bold] {result['run_id']}\n"
                f"[bold]Forked From Run:[/bold] {result['parent_run_id']} at Step {result['fork_at_step']}\n"
                f"[bold]Patches Applied:[/bold] {json.dumps(patch_dict)}\n"
                f"[bold]Total Steps in Branch:[/bold] {result['total_steps']}",
                title="[bold yellow]Branch Execution Result[/bold yellow]",
                border_style="magenta",
            )
        )
        return

    # Deterministic replay
    with console.status(f"[bold green]Replaying run {run_id} from cassette tapes...", spinner="dots"):
        result = replay_agent(run_id=run_id, storage=storage, strict=strict)

    if json_output:
        console.print_json(json.dumps(result, default=str))
        return

    console.print(
        Panel.fit(
            f"[bold green]✔ Deterministic Replay Successful (0 API Calls Made)[/bold green]\n"
            f"[bold]Original Run ID:[/bold] {result['run_id']}\n"
            f"[bold]Replayed Run ID:[/bold] {result['replay_id']}\n"
            f"[bold]Cassette Tool Calls Replayed:[/bold] {result['total_tool_calls']}",
            title="[bold cyan]VCR Replay Engine[/bold cyan]",
            border_style="green",
        )
    )


@app.command("fork")
def fork_command(
    run_id: str = typer.Option(..., "--run-id", "-r", help="UUID of historical run"),
    step: int = typer.Option(..., "--step", "-s", help="Step number to branch from"),
    patch: Optional[str] = typer.Option(None, "--patch", "-p", help="JSON dictionary to patch into state"),
    patch_tool: Optional[str] = typer.Option(None, "--patch-tool", help="JSON dictionary to patch into tool responses"),
    backend: str = typer.Option("file", "--backend", "-b", help="Storage backend (file or sqlite)"),
    storage_dir: str = typer.Option(".tapes", "--storage-dir", "-d", help="Tapes directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Time-travel debug: branch from step k, patch memory or tools, and evaluate recovery."""
    storage = _get_storage(backend, storage_dir)
    patch_state_dict = json.loads(patch) if patch else None
    patch_tool_dict = json.loads(patch_tool) if patch_tool else None

    with console.status(f"[bold magenta]Forking execution at step {step}...", spinner="dots"):
        result = fork_agent(
            parent_run_id=run_id,
            fork_at_step=step,
            patch_state=patch_state_dict,
            patch_tool_results=patch_tool_dict,
            storage=storage,
        )

    if json_output:
        console.print_json(json.dumps(result, default=str))
        return

    console.print(
        Panel.fit(
            f"[bold magenta]✔ Time-Travel Branch Created[/bold magenta]\n"
            f"[bold]Fork Run ID:[/bold] {result['run_id']}\n"
            f"[bold]Parent Run ID:[/bold] {result['parent_run_id']} (Forked at Step {result['fork_at_step']})\n"
            f"[bold]State Patches:[/bold] {json.dumps(patch_state_dict)}\n"
            f"[bold]Tool Patches:[/bold] {json.dumps(patch_tool_dict)}",
            title="[bold yellow]Time-Travel Debugger[/bold yellow]",
            border_style="magenta",
        )
    )

    # Show diff with parent
    diff = storage.diff_runs(run_id, result["run_id"])
    if diff.diverged_at_step is not None:
        console.print(f"[bold yellow]Divergence detected starting at Step {diff.diverged_at_step}[/bold yellow]")


@app.command("inspect")
def inspect_command(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run UUID to inspect"),
    backend: str = typer.Option("file", "--backend", "-b", help="Storage backend (file or sqlite)"),
    storage_dir: str = typer.Option(".tapes", "--storage-dir", "-d", help="Tapes directory"),
    step: Optional[int] = typer.Option(None, "--step", "-s", help="Inspect specific step snapshot"),
) -> None:
    """Inspect state snapshots, memory scratchpad, and recorded VCR cassette tapes."""
    storage = _get_storage(backend, storage_dir)
    manifest = storage.get_run_manifest(run_id)
    if not manifest:
        console.print(f"[bold red]Run '{run_id}' not found.[/bold red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]Run ID:[/bold] {manifest.run_id}\n"
            f"[bold]Parent Run ID:[/bold] {manifest.parent_run_id or 'None (Root Run)'}\n"
            f"[bold]Mode:[/bold] {manifest.mode.value} | [bold]Status:[/bold] {manifest.status}\n"
            f"[bold]Created:[/bold] {manifest.created_at.isoformat()}\n"
            f"[bold]Total Steps:[/bold] {manifest.total_steps} | [bold]Tool Calls:[/bold] {manifest.total_tool_calls}\n"
            f"[bold]Error:[/bold] {manifest.error or 'None'}",
            title="[bold blue]Run Manifest[/bold blue]",
        )
    )

    snapshots = storage.list_snapshots(run_id)
    if step is not None:
        target = next((s for s in snapshots if s.step == step), None)
        if target:
            console.print(f"[bold yellow]=== State Snapshot at Step {step} ({target.node_name}) ===[/bold yellow]")
            console.print_json(json.dumps(target.state_data, indent=2))
            if target.diff_from_parent:
                console.print("[bold cyan]Diff from Parent Snapshot:[/bold cyan]")
                console.print_json(json.dumps(target.diff_from_parent, indent=2))
        else:
            console.print(f"[red]No snapshot found for step {step}.[/red]")
        return

    table = Table(title="Step-by-Step State Timeline", border_style="cyan")
    table.add_column("Step", style="dim", width=6)
    table.add_column("Node", style="bold green")
    table.add_column("Status", style="yellow")
    table.add_column("Diff from Previous", style="magenta")
    for s in snapshots:
        diff_str = json.dumps(s.diff_from_parent) if s.diff_from_parent else "initial"
        if len(diff_str) > 60:
            diff_str = diff_str[:57] + "..."
        table.add_row(
            str(s.step),
            s.node_name,
            str(s.state_data.get("status", "N/A")),
            diff_str,
        )
    console.print(table)


@app.command("diff")
def diff_command(
    run_1_id: str = typer.Option(..., "--run-1", "-1", help="Base run ID"),
    run_2_id: str = typer.Option(..., "--run-2", "-2", help="Forked or comparison run ID"),
    backend: str = typer.Option("file", "--backend", "-b", help="Storage backend (file or sqlite)"),
    storage_dir: str = typer.Option(".tapes", "--storage-dir", "-d", help="Tapes directory"),
) -> None:
    """Compare two execution trajectories side-by-side to highlight state divergence."""
    storage = _get_storage(backend, storage_dir)
    diff = storage.diff_runs(run_1_id, run_2_id)

    console.print(
        Panel.fit(
            f"[bold]Run 1:[/bold] {run_1_id}\n"
            f"[bold]Run 2:[/bold] {run_2_id}\n"
            f"[bold]Divergence Step:[/bold] [bold red]{diff.diverged_at_step if diff.diverged_at_step is not None else 'None (Identical)'}[/bold red]",
            title="[bold yellow]Execution Trajectory Diff[/bold yellow]",
        )
    )

    table = Table(title="Step State Comparison", border_style="yellow")
    table.add_column("Step", width=6)
    table.add_column("Run 1 Node")
    table.add_column("Run 2 Node")
    table.add_column("Match?", justify="center")
    table.add_column("State Differences", style="dim")
    for comp in diff.step_comparisons:
        match_str = "[green]YES[/green]" if comp["match"] else "[red]NO[/red]"
        diff_str = json.dumps(comp["diff"]) if comp["diff"] else "-"
        if len(diff_str) > 70:
            diff_str = diff_str[:67] + "..."
        table.add_row(
            str(comp["step"]),
            comp["node_1"] or "-",
            comp["node_2"] or "-",
            match_str,
            diff_str,
        )
    console.print(table)


@app.command("list")
def list_command(
    backend: str = typer.Option("file", "--backend", "-b", help="Storage backend (file or sqlite)"),
    storage_dir: str = typer.Option(".tapes", "--storage-dir", "-d", help="Tapes directory"),
) -> None:
    """List all recorded and forked runs in storage."""
    storage = _get_storage(backend, storage_dir)
    runs = storage.list_runs()

    if not runs:
        console.print("[yellow]No runs recorded yet. Use 'agent-harness run' to execute a workflow.[/yellow]")
        return

    table = Table(title=f"Recorded Agent Runs ({len(runs)})", border_style="blue")
    table.add_column("Run ID", style="bold cyan")
    table.add_column("Mode", style="bold")
    table.add_column("Status", style="green")
    table.add_column("Forked From", style="dim")
    table.add_column("Steps", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Created At", style="dim")

    for r in runs:
        mode_style = "cyan" if r.mode == ExecutionMode.RECORD else ("green" if r.mode == ExecutionMode.REPLAY else "magenta")
        parent_str = f"{r.parent_run_id[:8]} (step {r.fork_step})" if r.parent_run_id else "-"
        table.add_row(
            r.run_id,
            f"[{mode_style}]{r.mode.value}[/{mode_style}]",
            r.status,
            parent_str,
            str(r.step_count),
            str(r.tool_call_count),
            r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )
    console.print(table)


@app.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto reload"),
) -> None:
    """Start the FastAPI microservice and interactive Time-Travel Debugging UI."""
    import uvicorn
    console.print(f"[bold green]Starting Agent Replay Harness Server on http://{host}:{port}...[/bold green]")
    uvicorn.run("agent_replay_harness.server.app:app", host=host, port=port, reload=reload)


def main():
    app()


if __name__ == "__main__":
    main()
