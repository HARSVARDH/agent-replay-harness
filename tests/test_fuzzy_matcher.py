"""Unit tests for fuzzy and semantic cassette matching."""

from agent_replay_harness.fuzzy_matcher import (
    compute_token_jaccard_similarity,
    fuzzy_match_record,
    normalize_value,
)
from agent_replay_harness.hashing import hash_tool_call
from agent_replay_harness.schemas import ToolCallRecord


def test_normalize_values():
    assert normalize_value("  TSLA   ") == "tsla"
    assert normalize_value(250.004) == 250.0
    assert normalize_value({"TICKER": "AAPL"}) == {"ticker": "aapl"}


def test_token_jaccard_similarity():
    sim = compute_token_jaccard_similarity("query=TSLA detailed=True", "detailed=True query=tsla")
    assert sim == 1.0

    low_sim = compute_token_jaccard_similarity("fetch TSLA", "fetch MSFT")
    assert low_sim < 0.5


def test_fuzzy_match_record():
    record = ToolCallRecord(
        run_id="run-1",
        step=1,
        tool_name="fetch_quote",
        args=["TSLA"],
        kwargs={"source": "AlphaVantage"},
        result={"price": 250.0},
        hash_key=hash_tool_call("fetch_quote", ["TSLA"], {"source": "AlphaVantage"}),
    )

    candidates = [record]

    # 1. Exact match
    match, score = fuzzy_match_record("fetch_quote", ["TSLA"], {"source": "AlphaVantage"}, candidates)
    assert match is not None
    assert score == 1.0

    # 2. Case difference: "tsla" instead of "TSLA"
    match2, score2 = fuzzy_match_record("fetch_quote", ["tsla"], {"source": "alphavantage"}, candidates)
    assert match2 is not None
    assert score2 >= 0.95
    assert match2.result["price"] == 250.0

    # 3. Unrelated tool
    no_match, _ = fuzzy_match_record("fetch_filings", ["TSLA"], {}, candidates)
    assert no_match is None
