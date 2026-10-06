"""Production crash-camera: exports and imports self-contained incident bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from agent_replay_harness.schemas import CassetteTape
from agent_replay_harness.storage.base import BaseTapeStorage
from agent_replay_harness.storage.file import FileTapeStorage


class CrashBundle(BaseModel):
    version: str = "1.0.0"
    incident_id: str
    tape: CassetteTape
    environment: dict = Field(default_factory=dict)


def export_crash_bundle(
    run_id: str,
    output_path: Optional[str | Path] = None,
    storage: Optional[BaseTapeStorage] = None,
) -> Path:
    """Package run manifest, events, snapshots, and cassettes into an incident file."""
    actual_storage = storage or FileTapeStorage()
    tape = actual_storage.get_tape(run_id)
    if not tape:
        raise ValueError(f"Run ID '{run_id}' not found in storage.")

    import platform
    import sys

    bundle = CrashBundle(
        incident_id=f"INCIDENT-{run_id[:8]}",
        tape=tape,
        environment={
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    )

    out_file = Path(output_path or f"{bundle.incident_id}.harness-crash")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(bundle.model_dump_json(indent=2))

    return out_file


def import_crash_bundle(
    bundle_path: str | Path,
    target_storage: Optional[BaseTapeStorage] = None,
) -> str:
    """Import a .harness-crash bundle into local storage for offline debugging."""
    actual_storage = target_storage or FileTapeStorage()
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Crash bundle '{bundle_path}' not found.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bundle = CrashBundle.model_validate(data)
    tape = bundle.tape

    # Hydrate manifest, events, snapshots, tool calls into storage
    actual_storage.create_run(tape.manifest)
    for ev in tape.events:
        actual_storage.append_event(ev)
    for snap in tape.snapshots:
        actual_storage.save_snapshot(snap)
    for tc in tape.tool_calls:
        actual_storage.record_tool_call(tc)

    return tape.manifest.run_id
