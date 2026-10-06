"""Agent state definitions for the financial risk and incident analysis workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class FinancialAgentState(TypedDict, total=False):
    """Memory footprint and scratchpad for the analysis agent."""
    query: str
    ticker: str
    step_count: int
    plan: List[str]
    market_data: Optional[Dict[str, Any]]
    filings_data: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    status: str  # "planning", "gathering", "analyzing", "recovering", "completed", "error"
    action_taken: Optional[str]
    final_report: Optional[Dict[str, Any]]
    error: Optional[str]
