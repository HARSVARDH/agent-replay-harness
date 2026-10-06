"""Deterministic hashing and canonical serialization utilities."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from typing import Any, Dict, List, Tuple, Union
from uuid import UUID

from pydantic import BaseModel


def default_serializer(obj: Any) -> Any:
    """Ensure non-standard Python types are serializable for canonical hashing."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, set):
        return sorted(list(obj), key=lambda x: str(x))
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def canonical_json_dumps(data: Any) -> str:
    """Serialize data to a canonical, deterministic JSON string with sorted keys."""
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=default_serializer,
    )


def hash_tool_call(
    tool_name: str,
    args: Union[Tuple[Any, ...], List[Any]] = (),
    kwargs: Dict[str, Any] | None = None,
) -> str:
    """Generate a deterministic 16-character SHA-256 hash for a tool call.

    Keyed by: tool_name + canonical(args) + canonical(kwargs).
    """
    clean_kwargs = kwargs or {}
    canonical_repr = (
        f"{tool_name}|"
        f"args={canonical_json_dumps(list(args))}|"
        f"kwargs={canonical_json_dumps(clean_kwargs)}"
    )
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()[:16]


def compute_dict_diff(old_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Compute added, removed, and modified fields between two dictionaries."""
    old_keys = set(old_dict.keys())
    new_keys = set(new_dict.keys())

    added = {k: new_dict[k] for k in new_keys - old_keys}
    removed = {k: old_dict[k] for k in old_keys - new_keys}
    modified: Dict[str, Dict[str, Any]] = {}

    for k in old_keys & new_keys:
        if old_dict[k] != new_dict[k]:
            modified[k] = {
                "old": old_dict[k],
                "new": new_dict[k],
            }

    diff: Dict[str, Any] = {}
    if added:
        diff["added"] = added
    if removed:
        diff["removed"] = removed
    if modified:
        diff["modified"] = modified

    return diff
