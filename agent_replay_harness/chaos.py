"""Chaos engineering and automated resilience evaluation matrix for agent workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from agent_replay_harness.runner import fork_agent, run_agent
from agent_replay_harness.schemas import ExecutionMode
from agent_replay_harness.storage.base import BaseTapeStorage
from agent_replay_harness.storage.file import FileTapeStorage

console = Console()


class ChaosScenarioResult(BaseModel):
    name: str
    fault_type: str
    target_tool_or_state: str
    fork_step: int
    recovered: bool
    recovery_action: Optional[str] = None
    error_message: Optional[str] = None
    fork_run_id: str


class ChaosReport(BaseModel):
    parent_run_id: str
    total_scenarios: int
    passed_scenarios: int
    resilience_score_pct: float
    scenarios: List[ChaosScenarioResult] = Field(default_factory=list)


CHAOS_SCENARIOS = [
    {
        "name": "Market Circuit Breaker",
        "fault_type": "STATE_ANOMALY",
        "target": "market_data",
        "step": 3,
        "patch_state": {
            "market_data": {
                "status": "circuit_breaker_triggered",
                "error": "Flash crash detected (-12% price plunge)",
            }
        },
        "patch_tool": None,
    },
    {
        "name": "SEC Edgar HTTP 504 Timeout",
        "fault_type": "TOOL_ERROR",
        "target": "fetch_company_filing",
        "step": 2,
        "patch_state": None,
        "patch_tool": {"fetch_company_filing": {"error": "HTTP 504 Gateway Timeout: EDGAR unavailable"}},
    },
    {
        "name": "Quantitative Risk Model 500",
        "fault_type": "TOOL_ERROR",
        "target": "evaluate_risk_model",
        "step": 3,
        "patch_state": None,
        "patch_tool": {"evaluate_risk_model": {"error": "InternalServerError: Matrix singular decomposition failure"}},
    },
    {
        "name": "Generic Tool Rate Limit 429",
        "fault_type": "TOOL_ERROR",
        "target": "tool_result",
        "step": 2,
        "patch_state": None,
        "patch_tool": {"tool_result": "Error 429: Too Many Requests - Token quota exhausted"},
    },
]


def run_chaos_matrix(
    query: str = "Analyze TSLA financial risk and filings",
    storage: Optional[BaseTapeStorage] = None,
    scenarios: Optional[List[Dict[str, Any]]] = None,
) -> ChaosReport:
    """Run automated resilience testing suite, injecting faults and measuring agent recovery."""
    actual_storage = storage or FileTapeStorage()
    active_scenarios = scenarios or CHAOS_SCENARIOS

    # 1. Establish baseline run in RECORD mode
    baseline = run_agent(query=query, mode=ExecutionMode.RECORD, storage=actual_storage)
    parent_id = baseline["run_id"]

    results: List[ChaosScenarioResult] = []

    for sc in active_scenarios:
        name = sc["name"]
        step = sc["step"]
        patch_state = sc.get("patch_state")
        patch_tool = sc.get("patch_tool")

        try:
            fork_res = fork_agent(
                parent_run_id=parent_id,
                fork_at_step=step,
                patch_state=patch_state,
                patch_tool_results=patch_tool,
                storage=actual_storage,
            )
            out = fork_res.get("output", {})
            action = out.get("action_taken")
            recovered = (
                out.get("status") in ["completed", "remediated"]
                and (action is not None or out.get("error") is not None)
            )
            results.append(
                ChaosScenarioResult(
                    name=name,
                    fault_type=sc["fault_type"],
                    target_tool_or_state=sc["target"],
                    fork_step=step,
                    recovered=recovered,
                    recovery_action=action,
                    fork_run_id=fork_res["run_id"],
                )
            )
        except Exception as exc:
            results.append(
                ChaosScenarioResult(
                    name=name,
                    fault_type=sc["fault_type"],
                    target_tool_or_state=sc["target"],
                    fork_step=step,
                    recovered=False,
                    error_message=str(exc),
                    fork_run_id="failed",
                )
            )

    passed = sum(1 for r in results if r.recovered)
    total = len(results)
    score = round((passed / total) * 100, 1) if total > 0 else 0.0

    return ChaosReport(
        parent_run_id=parent_id,
        total_scenarios=total,
        passed_scenarios=passed,
        resilience_score_pct=score,
        scenarios=results,
    )


def print_chaos_report(report: ChaosReport) -> None:
    """Print beautifully formatted Rich table for CI/CD logs."""
    table = Table(
        title=f"Agent Chaos Resilience Matrix — Score: {report.resilience_score_pct}% ({report.passed_scenarios}/{report.total_scenarios} Recovered)",
        border_style="magenta",
    )
    table.add_column("Scenario", style="bold white")
    table.add_column("Target", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Outcome", justify="center")
    table.add_column("Recovery Action Taken", style="dim")

    for s in report.scenarios:
        outcome_str = "[bold green]PASS (Healed)[/bold green]" if s.recovered else "[bold red]FAIL (Crash)[/bold red]"
        table.add_row(
            s.name,
            s.target_tool_or_state,
            s.fault_type,
            outcome_str,
            s.recovery_action or s.error_message or "None",
        )

    console.print(table)
