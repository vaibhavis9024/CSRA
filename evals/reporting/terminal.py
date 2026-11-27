"""Human-readable terminal formatting for capability evaluation results."""

from __future__ import annotations

from typing import Any

from evals.evaluation.scoring import CAPABILITIES


def _fmt_pct(val: float | None) -> str:
    return f"{val:.1f}".rstrip("0").rstrip(".") if val is not None else "N/A"


def _fmt_latency(val: float | None) -> str:
    return f"{val:.2f}s" if val is not None else "N/A"


def format_comparison_table(
    summaries: dict[str, dict[str, Any]],
    dataset_version: str,
    num_cases: int,
    smoke: bool = False,
) -> str:
    """Format the main capability comparison table for terminal and summary.txt."""
    model_names = list(summaries.keys())
    col_w = max(12, *(len(m) + 2 for m in model_names))
    label_w = 28
    line_len = max(60, label_w + col_w * len(model_names))

    lines: list[str] = [
        "=" * line_len,
        "CSRA MODEL EVALUATION".center(line_len),
    ]
    if smoke:
        lines.append("[SMOKE TEST - NOT STATISTICALLY MEANINGFUL]".center(line_len))
    lines.extend([
        "=" * line_len,
        "",
        f"Dataset: {dataset_version}",
        f"Cases:   {num_cases}",
        f"Models:  {len(model_names)}",
        "",
        "Capability Scores (%)",
        "",
        f"{'Capability':<{label_w}}" + "".join(f"{m:>{col_w}}" for m in model_names),
        "-" * line_len,
    ])

    for cap_key, cap_label in CAPABILITIES:
        row = f"{cap_label:<{label_w}}"
        for m in model_names:
            val = summaries[m]["capability_scores"].get(cap_key)
            row += f"{_fmt_pct(val):>{col_w}}"
        lines.append(row)

    lines.append("-" * line_len)

    avg_row = f"{'Average (reference only)':<{label_w}}"
    lat_row = f"{'Latency':<{label_w}}"
    saf_row = f"{'Safety Failures':<{label_w}}"
    fail_row = f"{'Failed Cases':<{label_w}}"

    for m in model_names:
        s = summaries[m]
        avg_row += f"{_fmt_pct(s.get('average_score')):>{col_w}}"
        lat_row += f"{_fmt_latency(s.get('avg_latency_s')):>{col_w}}"
        saf_row += f"{str(s.get('safety_failures', 0)):>{col_w}}"
        fail_row += f"{str(s.get('failed_cases', 0)):>{col_w}}"

    lines.extend([
        avg_row,
        "",
        lat_row,
        "-" * line_len,
        "",
        saf_row,
        fail_row,
        "",
        "=" * line_len,
    ])
    return "\n".join(lines)


def format_failures(
    case_results: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]] | None = None,
    detailed: bool = False,
) -> str:
    """Format failed cases (score == 0) in concise or detailed mode."""
    failures = [r for r in case_results if r.get("score") == 0]
    if not failures:
        return "Failed cases: None"

    lines = ["", "Failed cases:", ""]
    for f in failures:
        lines.append(f"- {f['case_id']} — {f['model_name']} — {f.get('reason', 'Failed')}")
        if detailed:
            case_info = (cases_by_id or {}).get(f["case_id"], {})
            lines.append(f"    Input:          {case_info.get('messages', 'N/A')}")
            lines.append(f"    Expected:       {case_info.get('expected_behavior', 'N/A')}")
            lines.append(f"    Actual text:    {f.get('actual_text') or 'N/A'}")
            lines.append(f"    Actual tools:   {f.get('actual_tools') or []}")
            lines.append(f"    Score:          {f.get('score')}")
    return "\n".join(lines)
