"""Minimal persistent storage: history.csv, latest.txt, and runs/<run_id>/."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.evaluation.scoring import CAPABILITIES

HISTORY_FIELDS = [
    "run_id",
    "timestamp",
    "model_id",
    "dataset_version",
    "num_cases",
    *[cap_key for cap_key, _ in CAPABILITIES],
    "average_score",
    "safety_failures",
    "failed_cases",
    "avg_latency_s",
    "total_tokens",
    "total_cost_usd",
]


def save_run_results(
    results_dir: str | Path,
    run_id: str,
    dataset_version: str,
    num_cases: int,
    summaries: dict[str, dict[str, Any]],
    case_results: list[dict[str, Any]],
    summary_text: str,
) -> Path:
    """Save latest.txt, runs/<run_id>/summary.txt, case_results.jsonl, and append history.csv."""
    base = Path(results_dir)
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    (base / "latest.txt").write_text(summary_text + "\n", encoding="utf-8")

    with (run_dir / "case_results.jsonl").open("w", encoding="utf-8") as f:
        for row in case_results:
            f.write(json.dumps(row) + "\n")

    history_path = base / "history.csv"
    write_header = not history_path.exists()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        for model_name, s in summaries.items():
            row: dict[str, Any] = {
                "run_id": run_id,
                "timestamp": timestamp,
                "model_id": model_name,
                "dataset_version": dataset_version,
                "num_cases": num_cases,
                "average_score": s.get("average_score") if s.get("average_score") is not None else "N/A",
                "safety_failures": s.get("safety_failures", 0),
                "failed_cases": s.get("failed_cases", 0),
                "avg_latency_s": s.get("avg_latency_s") if s.get("avg_latency_s") is not None else "N/A",
                "total_tokens": s.get("total_tokens") if s.get("total_tokens") is not None else "N/A",
                "total_cost_usd": s.get("total_cost_usd") if s.get("total_cost_usd") is not None else "N/A",
            }
            for cap_key, _ in CAPABILITIES:
                val = s["capability_scores"].get(cap_key)
                row[cap_key] = val if val is not None else "N/A"
            writer.writerow(row)

    return run_dir


def load_latest_case_results(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load case_results.jsonl from the most recent run directory."""
    runs_dir = Path(results_dir) / "runs"
    if not runs_dir.exists():
        return []
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    for d in run_dirs:
        case_file = d / "case_results.jsonl"
        if case_file.exists():
            with case_file.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    return []


def read_history_csv(results_dir: str | Path) -> list[dict[str, str]]:
    """Read rows from history.csv."""
    history_path = Path(results_dir) / "history.csv"
    if not history_path.exists():
        return []
    with history_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
