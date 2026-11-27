"""CLI runner for CSRA capability-based model evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.evaluation.evaluators import evaluate_case
from evals.evaluation.scoring import aggregate_model_summary
from evals.models.registry import ModelRegistry, load_evaluation_config
from evals.reporting.storage import (
    load_latest_case_results,
    read_history_csv,
    save_run_results,
)
from evals.reporting.terminal import format_comparison_table, format_failures


def load_cases(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Load capability evaluation cases from a JSONL file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_num}: {exc}") from exc
    return cases


def select_smoke_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick 1 case per capability for a quick smoke test."""
    seen_caps: set[str] = set()
    smoke: list[dict[str, Any]] = []
    for c in cases:
        cap = c["capability"]
        if cap not in seen_caps:
            seen_caps.add(cap)
            smoke.append(c)
    return smoke


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="CSRA Capability Model Evaluation Runner")
    parser.add_argument("--config", default="configs/evaluation.yaml", help="Path to evaluation YAML")
    parser.add_argument("--model", default=None, help="Evaluate only a single configured model")
    parser.add_argument("--smoke", action="store_true", help="Run 1 case per capability (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate run without making API calls")
    parser.add_argument("--show-failures", action="store_true", help="Display failed cases from latest run")
    parser.add_argument("--detailed", action="store_true", help="Show detailed failure info with --show-failures")
    parser.add_argument("--history", action="store_true", help="Display historical runs from history.csv")
    return parser.parse_args(argv)


async def run_async(args: argparse.Namespace) -> int:
    """Execute the evaluation or reporting command."""
    load_dotenv()
    config = load_evaluation_config(args.config)
    cases = load_cases(config.dataset_path)
    cases_by_id = {c["case_id"]: c for c in cases}

    if args.history:
        rows = read_history_csv(config.results_dir)
        if not rows:
            print("No historical runs found.")
            return 0
        print(f"{'Run ID':<18} {'Model':<22} {'Avg':>6} {'SafeFail':>9} {'Failed':>7} {'Latency':>8}")
        print("-" * 74)
        for r in rows:
            print(
                f"{r['run_id']:<18} {r['model_id']:<22} {r['average_score']:>6} "
                f"{r['safety_failures']:>9} {r['failed_cases']:>7} {r['avg_latency_s']:>8}"
            )
        return 0

    if args.show_failures:
        latest = load_latest_case_results(config.results_dir)
        if not latest:
            print("No previous run results found.")
            return 0
        print(format_failures(latest, cases_by_id=cases_by_id, detailed=args.detailed))
        return 0

    if args.smoke:
        cases = select_smoke_cases(cases)

    registry = ModelRegistry(config)
    models = [registry.get_model(args.model)] if args.model else registry.enabled_models()
    if not models:
        print("No enabled models configured.")
        return 1

    judge_adapter = None
    if config.judge.enabled and not args.dry_run:
        judge_adapter = registry.create_adapter(
            model_id=config.judge.model_id,
            model_name="judge",
            require_api_key=True,
        )

    all_case_results: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    semaphore = asyncio.Semaphore(config.concurrency)

    try:
        for m in models:
            adapter = registry.create_adapter(
                model_id=m.id,
                model_name=m.name,
                require_api_key=not args.dry_run,
                temperature=m.temperature,
                max_tokens=m.max_tokens,
            )
            try:
                async def _run_one(c: dict[str, Any]) -> dict[str, Any]:
                    async with semaphore:
                        return await evaluate_case(c, adapter, judge_adapter, dry_run=args.dry_run)

                model_results = await asyncio.gather(*[_run_one(c) for c in cases])
            finally:
                await adapter.aclose()

            all_case_results.extend(model_results)
            summaries[m.name] = aggregate_model_summary(model_results)
    finally:
        if judge_adapter is not None:
            await judge_adapter.aclose()

    summary_text = format_comparison_table(
        summaries=summaries,
        dataset_version=config.dataset_version,
        num_cases=len(cases),
        smoke=args.smoke,
    )
    print(summary_text)
    print(format_failures(all_case_results, cases_by_id=cases_by_id, detailed=args.detailed))

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = save_run_results(
        results_dir=config.results_dir,
        run_id=run_id,
        dataset_version=config.dataset_version,
        num_cases=len(cases),
        summaries=summaries,
        case_results=all_case_results,
        summary_text=summary_text,
    )
    print(f"\nSaved run to: {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    return asyncio.run(run_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
