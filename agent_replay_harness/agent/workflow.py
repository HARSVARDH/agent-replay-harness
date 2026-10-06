"""LangGraph StateGraph workflow for financial analysis with fault recovery."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agent_replay_harness.agent.state import FinancialAgentState
from agent_replay_harness.agent.tools import (
    evaluate_risk_model,
    execute_circuit_breaker_hedge,
    fetch_company_filing,
    fetch_market_quote,
    publish_report,
)


def planner_node(state: FinancialAgentState) -> Dict[str, Any]:
    """Parse query, resolve ticker, and initialize multi-step plan."""
    query = state.get("query", "Analyze TSLA risk and performance")
    ticker_match = re.search(r"\b([A-Z]{2,5})\b", query)
    ticker = ticker_match.group(1) if ticker_match else "TSLA"

    return {
        "ticker": ticker,
        "plan": ["fetch_market_quote", "fetch_company_filing", "evaluate_risk_model", "publish_report"],
        "status": "gathering",
        "step_count": (state.get("step_count") or 0) + 1,
    }


def gather_data_node(state: FinancialAgentState) -> Dict[str, Any]:
    """Execute data gathering tools with resilient exception handling."""
    ticker = state.get("ticker", "TSLA")

    try:
        quote = fetch_market_quote(ticker=ticker)
    except Exception as exc:
        quote = {"ticker": ticker, "status": "circuit_breaker_triggered", "error": f"Market feed failure: {exc}"}

    try:
        filing = fetch_company_filing(ticker=ticker, form_type="10-K")
    except Exception as exc:
        filing = {"ticker": ticker, "financials": {"revenue_billions": 50.0}, "error": f"Filings service failure: {exc}"}

    if isinstance(quote, str):
        quote = {"ticker": ticker, "status": "circuit_breaker_triggered", "error": quote}
    if isinstance(filing, str):
        filing = {"ticker": ticker, "financials": {"revenue_billions": 50.0}, "error": filing}

    err = (quote.get("error") if isinstance(quote, dict) else str(quote)) or (
        filing.get("error") if isinstance(filing, dict) else str(filing)
    )
    status = "recovery_needed" if err else "analyzing"

    return {
        "market_data": quote,
        "filings_data": filing,
        "status": status,
        "error": err,
        "step_count": (state.get("step_count") or 0) + 1,
    }


def analyze_risk_node(state: FinancialAgentState) -> Dict[str, Any]:
    """Run quantitative model and detect anomalies or circuit breakers."""
    ticker = state.get("ticker", "TSLA")
    market = state.get("market_data") or {}
    filings = state.get("filings_data") or {}

    if isinstance(market, str):
        market = {"error": market}
    if isinstance(filings, str):
        filings = {"error": filings}

    # Check for fault injection or market anomalies in state
    if market.get("status") == "circuit_breaker_triggered" or market.get("error") or filings.get("error") or state.get("error"):
        err_msg = market.get("error") or filings.get("error") or state.get("error") or "Market circuit breaker triggered"
        return {
            "status": "recovery_needed",
            "error": err_msg,
            "step_count": (state.get("step_count") or 0) + 1,
        }

    price = market.get("price", 100.0)
    rev = filings.get("financials", {}).get("revenue_billions", 50.0)

    # Run virtualized risk tool
    try:
        risk_result = evaluate_risk_model(ticker=ticker, price=price, revenue_billions=rev)
        return {
            "risk_assessment": risk_result,
            "status": "reporting",
            "step_count": (state.get("step_count") or 0) + 1,
        }
    except Exception as exc:
        return {
            "status": "recovery_needed",
            "error": f"Risk model execution failure: {exc}",
            "step_count": (state.get("step_count") or 0) + 1,
        }


def route_analysis(state: FinancialAgentState) -> str:
    """Conditional router: branch to recovery node if anomaly detected, else reporting."""
    if state.get("status") == "recovery_needed" or state.get("error"):
        return "recovery"
    return "reporting"


def recovery_node(state: FinancialAgentState) -> Dict[str, Any]:
    """Execute remediation protocol when anomalous condition or error is active."""
    ticker = state.get("ticker", "TSLA")
    reason = state.get("error", "Anomalous market deviation detected")

    # Run remediation tool
    try:
        hedge = execute_circuit_breaker_hedge(ticker=ticker, reason=reason)
    except Exception as exc:
        hedge = f"Emergency circuit breaker tripped: {exc}"

    action_desc = hedge.get("action") if isinstance(hedge, dict) else str(hedge)

    return {
        "action_taken": f"Remediation hedge executed: {action_desc}",
        "status": "remediated",
        "step_count": (state.get("step_count") or 0) + 1,
    }


def reporting_node(state: FinancialAgentState) -> Dict[str, Any]:
    """Compile final analyst dossier and publish via notification stream."""
    ticker = state.get("ticker", "TSLA")
    summary = {
        "ticker": ticker,
        "market": state.get("market_data"),
        "filings": state.get("filings_data"),
        "risk": state.get("risk_assessment"),
        "remediation": state.get("action_taken"),
        "error_recovered": state.get("error"),
    }

    # Call virtualized publishing tool
    pub = publish_report(ticker=ticker, summary=summary)

    return {
        "final_report": {
            "dossier": summary,
            "publishing_receipt": pub,
        },
        "status": "completed",
        "step_count": (state.get("step_count") or 0) + 1,
    }


def build_agent_graph(checkpointer: Optional[BaseCheckpointSaver] = None) -> Any:
    """Construct and compile the LangGraph StateGraph with checkpointer integration."""
    builder = StateGraph(FinancialAgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("gather_data", gather_data_node)
    builder.add_node("analyze_risk", analyze_risk_node)
    builder.add_node("recovery", recovery_node)
    builder.add_node("reporting", reporting_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "gather_data")
    builder.add_edge("gather_data", "analyze_risk")

    builder.add_conditional_edges(
        "analyze_risk",
        route_analysis,
        {
            "recovery": "recovery",
            "reporting": "reporting",
        },
    )

    builder.add_edge("recovery", "reporting")
    builder.add_edge("reporting", END)

    return builder.compile(checkpointer=checkpointer)
