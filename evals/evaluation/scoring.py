"""0-2 capability scoring and aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

CAPABILITIES: list[tuple[str, str]] = [
    ("intent_understanding", "Intent Understanding"),
    ("information_extraction", "Information Extraction"),
    ("tool_selection", "Tool Selection"),
    ("multi_step_reasoning", "Multi-step Reasoning"),
    ("customer_communication", "Customer Communication"),
    ("knowledge_retrieval", "Knowledge Retrieval"),
    ("context_handling", "Context Handling"),
    ("safety_compliance", "Safety & Compliance"),
]


def capability_percentage(scores: list[int | None]) -> float | None:
    """Calculate Capability Score (%) = (earned / max_possible) * 100.

    Returns None when no valid scores are available (displayed as N/A).
    """
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    max_points = 2 * len(valid)
    return round((sum(valid) / max_points) * 100.0, 1)


def aggregate_model_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate 0-2 case results for a single model into capability percentages and stats."""
    by_cap: dict[str, list[int | None]] = defaultdict(list)
    latencies: list[float] = []
    tokens: list[int] = []
    costs: list[float] = []
    safety_failures = 0
    failed_cases = 0

    for row in case_results:
        cap = row["capability"]
        score = row.get("score")
        by_cap[cap].append(score)

        if score == 0:
            failed_cases += 1
            if cap == "safety_compliance":
                safety_failures += 1

        if row.get("error") is None and row.get("latency_ms") is not None:
            latencies.append(float(row["latency_ms"]))
        if row.get("total_tokens") is not None:
            tokens.append(int(row["total_tokens"]))
        if row.get("cost_usd") is not None:
            costs.append(float(row["cost_usd"]))

    cap_scores: dict[str, float | None] = {
        cap_key: capability_percentage(by_cap.get(cap_key, []))
        for cap_key, _ in CAPABILITIES
    }
    valid_cap_scores = [v for v in cap_scores.values() if v is not None]
    avg_score = round(sum(valid_cap_scores) / len(valid_cap_scores), 1) if valid_cap_scores else None
    avg_latency_s = round((sum(latencies) / len(latencies)) / 1000.0, 2) if latencies else None

    return {
        "capability_scores": cap_scores,
        "average_score": avg_score,
        "avg_latency_s": avg_latency_s,
        "safety_failures": safety_failures,
        "failed_cases": failed_cases,
        "total_cases": len(case_results),
        "total_tokens": sum(tokens) if tokens else None,
        "total_cost_usd": round(sum(costs), 6) if costs else None,
    }
