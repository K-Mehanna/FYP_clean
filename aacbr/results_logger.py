"""Utilities for logging AACBR evaluation results to a shared leaderboard.json."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

_DEFAULT_LEADERBOARD = Path(__file__).resolve().parent / "leaderboard.json"


def log_result(
    entry: dict,
    leaderboard_path: str | Path | None = None,
) -> Path:
    """Append *entry* to the leaderboard JSON, returning the path written to.

    A ``run_id`` (UUID4) and ``timestamp`` (ISO-8601) are injected automatically
    unless already present in *entry*.
    """
    path = Path(leaderboard_path) if leaderboard_path else _DEFAULT_LEADERBOARD

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        records = []

    record = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    record.update(entry)

    records.append(record)

    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, path)

    print(f"Result logged to {path}  (run_id={record['run_id']})")
    return path


def load_leaderboard(path: str | Path | None = None) -> pd.DataFrame:
    """Load leaderboard.json and return a flat pandas DataFrame."""
    path = Path(path) if path else _DEFAULT_LEADERBOARD
    if not path.exists():
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return pd.json_normalize(records)
