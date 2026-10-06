"""External tools wrapped with @record_tool_call for virtualization."""

from __future__ import annotations

from datetime import datetime, timezone
import random
import time
from typing import Any, Dict

from agent_replay_harness.interceptor import record_tool_call


@record_tool_call("fetch_market_quote")
def fetch_market_quote(ticker: str) -> Dict[str, Any]:
    """Fetch live ticker market quote (simulates non-deterministic API call)."""
    # Simulate API latency
    time.sleep(0.05)
    # Ticker baseline pricing
    baselines = {
        "AAPL": 225.50,
        "MSFT": 440.00,
        "NVDA": 130.00,
        "TSLA": 250.00,
        "GOOGL": 180.00,
    }
    base = baselines.get(ticker.upper(), 100.0)
    # Simulate jitter / non-determinism in live mode
    jitter = round(random.uniform(-1.5, 1.5), 2)
    price = round(base + jitter, 2)

    return {
        "ticker": ticker.upper(),
        "price": price,
        "currency": "USD",
        "volume": int(random.uniform(5_000_000, 20_000_000)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "MockLiveMarketDataFeed",
    }


@record_tool_call("fetch_company_filing")
def fetch_company_filing(ticker: str, form_type: str = "10-K") -> Dict[str, Any]:
    """Fetch company SEC financial filing data."""
    time.sleep(0.04)
    filings = {
        "AAPL": {"revenue_billions": 383.2, "net_margin_pct": 25.3, "debt_to_equity": 1.4},
        "MSFT": {"revenue_billions": 245.1, "net_margin_pct": 36.1, "debt_to_equity": 0.4},
        "NVDA": {"revenue_billions": 60.9, "net_margin_pct": 48.8, "debt_to_equity": 0.2},
        "TSLA": {"revenue_billions": 96.7, "net_margin_pct": 15.5, "debt_to_equity": 0.1},
        "GOOGL": {"revenue_billions": 307.4, "net_margin_pct": 24.0, "debt_to_equity": 0.1},
    }
    data = filings.get(
        ticker.upper(),
        {"revenue_billions": 50.0, "net_margin_pct": 12.0, "debt_to_equity": 0.8},
    )
    return {
        "ticker": ticker.upper(),
        "form_type": form_type,
        "financials": data,
        "filing_status": "VERIFIED_AUDITED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@record_tool_call("evaluate_risk_model")
def evaluate_risk_model(ticker: str, price: float, revenue_billions: float) -> Dict[str, Any]:
    """Run quantitative financial risk evaluation model."""
    time.sleep(0.03)
    # Simulated risk evaluation
    pe_ratio = round(price / max(revenue_billions * 0.1, 1.0), 2)
    volatility = round(0.15 + (0.01 * (pe_ratio % 5)), 3)
    var_95 = round(price * volatility * 1.65, 2)

    risk_rating = "LOW"
    if volatility > 0.25 or pe_ratio > 40:
        risk_rating = "HIGH"
    elif volatility > 0.18:
        risk_rating = "MEDIUM"

    return {
        "ticker": ticker.upper(),
        "pe_ratio": pe_ratio,
        "annualized_volatility": volatility,
        "value_at_risk_95": var_95,
        "risk_rating": risk_rating,
        "recommendation": "HOLD" if risk_rating == "HIGH" else "ACCUMULATE",
        "model_version": "QuantRiskEngine-v2.4",
    }


@record_tool_call("publish_report")
def publish_report(ticker: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Publish the final analyst evaluation report to subscriber stream."""
    time.sleep(0.02)
    return {
        "ticker": ticker.upper(),
        "report_id": f"REP-{ticker.upper()}-{int(time.time())}",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "status": "DELIVERED",
        "distribution_channel": "AnalystPortal & SlackWebhook",
    }


@record_tool_call("execute_circuit_breaker_hedge")
def execute_circuit_breaker_hedge(ticker: str, reason: str) -> Dict[str, Any]:
    """Remediation tool triggered when an anomaly, fault, or critical risk occurs."""
    time.sleep(0.02)
    return {
        "ticker": ticker.upper(),
        "hedge_status": "EXECUTED",
        "action": "HALT_ALLOCATION_AND_REDUCE_EXPOSURE",
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
