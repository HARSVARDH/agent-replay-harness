"""Fuzzy and semantic cassette matcher for resilient LLM tool call replay."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agent_replay_harness.hashing import canonical_json_dumps, hash_tool_call
from agent_replay_harness.schemas import ToolCallRecord


def normalize_value(val: Any) -> Any:
    """Normalize strings, numbers, and collections for resilient comparison."""
    if isinstance(val, str):
        # Lowercase, collapse multiple spaces, strip punctuation
        cleaned = re.sub(r"\s+", " ", val.strip().lower())
        return cleaned
    if isinstance(val, float):
        return round(val, 2)
    if isinstance(val, dict):
        return {k.lower(): normalize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [normalize_value(x) for x in val]
    return val


def compute_token_jaccard_similarity(str1: str, str2: str) -> float:
    """Compute token-level Jaccard similarity between two serialized strings."""
    tokens1 = set(re.findall(r"\w+", str1.lower()))
    tokens2 = set(re.findall(r"\w+", str2.lower()))
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def fuzzy_match_record(
    tool_name: str,
    args: List[Any],
    kwargs: Dict[str, Any],
    candidate_records: List[ToolCallRecord],
    threshold: float = 0.75,
) -> Tuple[Optional[ToolCallRecord], float]:
    """Find the best matching cassette record using normalized structural & token similarity.

    Returns:
        (best_record, similarity_score)
    """
    matching_tools = [r for r in candidate_records if r.tool_name == tool_name]
    if not matching_tools:
        return None, 0.0

    # 1. Exact canonical hash match (Fast Path)
    exact_hash = hash_tool_call(tool_name, args, kwargs)
    for record in matching_tools:
        if record.hash_key == exact_hash:
            return record, 1.0

    # 2. Normalized comparison
    norm_args = normalize_value(list(args))
    norm_kwargs = normalize_value(kwargs)
    canonical_norm_input = canonical_json_dumps({"args": norm_args, "kwargs": norm_kwargs})

    best_record: Optional[ToolCallRecord] = None
    best_score = 0.0

    for record in matching_tools:
        cand_norm_args = normalize_value(record.args)
        cand_norm_kwargs = normalize_value(record.kwargs)

        # Check structural equality after normalization
        if norm_args == cand_norm_args and norm_kwargs == cand_norm_kwargs:
            return record, 0.98

        # Compute fuzzy token Jaccard similarity
        cand_repr = canonical_json_dumps({"args": cand_norm_args, "kwargs": cand_norm_kwargs})
        score = compute_token_jaccard_similarity(canonical_norm_input, cand_repr)

        if score > best_score:
            best_score = score
            best_record = record

    if best_score >= threshold and best_record is not None:
        return best_record, best_score

    return None, best_score
