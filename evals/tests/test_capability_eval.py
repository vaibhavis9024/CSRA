"""Unit tests for dataset loading, capability scoring, evaluators, judge, and reporting."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from evals.evaluation.evaluators import score_deterministic
from evals.evaluation.judge import judge_response
from evals.evaluation.scoring import aggregate_model_summary, capability_percentage
from evals.models.base import NormalizedResponse, ToolCall
from evals.models.cloudflare import CloudflareAdapter, build_cloudflare_base_url
from evals.reporting.storage import read_history_csv, save_run_results
from evals.reporting.terminal import format_comparison_table, format_failures
from evals.run_evaluation import load_cases, main as cli_main, select_smoke_cases


def test_capability_dataset_has_40_cases_across_8_capabilities() -> None:
    cases = load_cases("data/evals/capability_cases.jsonl")
    assert len(cases) == 40
    smoke = select_smoke_cases(cases)
    assert len(smoke) == 8


def test_capability_scoring_and_missing_metrics() -> None:
    assert capability_percentage([2, 1, 2]) == 83.3
    assert capability_percentage([None, None]) is None

    summary = aggregate_model_summary([
        {"capability": "intent_understanding", "score": 2, "latency_ms": 1000.0, "error": None},
        {"capability": "safety_compliance", "score": 0, "latency_ms": 2000.0, "error": None},
    ])
    assert summary["capability_scores"]["intent_understanding"] == 100.0
    assert summary["capability_scores"]["tool_selection"] is None
    assert summary["safety_failures"] == 1
    assert summary["failed_cases"] == 1
    assert summary["avg_latency_s"] == 1.5


def test_deterministic_evaluators_and_safety_detection() -> None:
    intent_case = {"case_id": "INT-02", "capability": "intent_understanding", "expected_output": {"intents": ["billing", "delivery"]}}
    resp_ok = NormalizedResponse(model_name="m", provider="openrouter", model_id="id", structured_output={"intents": ["billing", "delivery"]})
    assert score_deterministic(intent_case, resp_ok)["score"] == 2

    safety_case = {"case_id": "SAFE-01", "capability": "safety_compliance", "expected_output": {"forbidden_tools": ["request_refund"], "must_escalate": False}}
    resp_unsafe = NormalizedResponse(
        model_name="m", provider="openrouter", model_id="id",
        tool_calls=[ToolCall(name="request_refund", arguments={"order_id": "ORD-1", "amount": 5000})],
    )
    assert score_deterministic(safety_case, resp_unsafe)["score"] == 0


@pytest.mark.asyncio
async def test_llm_judge_parsing_and_self_judge_prevention() -> None:
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(200, json={"choices": [{"message": {"content": '{"score": 2, "reason": "Clear and accurate."}'}}]})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        judge = CloudflareAdapter(
            model_name="judge",
            model_id="@cf/meta/llama-3.1-8b-instruct",
            api_key="k",
            base_url=build_cloudflare_base_url("acc123"),
            client=client,
        )
        case = {"case_id": "COM-01", "capability": "customer_communication", "messages": [], "expected_behavior": "Empathetic"}

        skipped = await judge_response(judge, "@cf/meta/llama-3.1-8b-instruct", case, "Hi")
        assert skipped["score"] is None

        scored = await judge_response(judge, "@cf/meta/llama-3.3-70b-instruct-fp8-fast", case, "Sorry about the delay!")
        assert scored["score"] == 2
        assert scored["judged_by_llm"] is True


def test_terminal_table_and_minimal_storage(tmp_path: Path) -> None:
    summaries = {
        "Model A": {
            "capability_scores": {
                "intent_understanding": 100.0, "information_extraction": 90.0,
                "tool_selection": None, "multi_step_reasoning": 80.0,
                "customer_communication": 90.0, "knowledge_retrieval": 85.0,
                "context_handling": 80.0, "safety_compliance": 100.0,
            },
            "average_score": 89.3, "avg_latency_s": 1.25, "safety_failures": 0,
            "failed_cases": 1, "total_tokens": 500, "total_cost_usd": 0.001,
        }
    }
    table = format_comparison_table(summaries, "csra_capability_v1", 8, smoke=True)
    assert "SMOKE TEST" in table
    assert "N/A" in table

    run_dir = save_run_results(
        tmp_path, "run_001", "csra_capability_v1", 8, summaries,
        [{"case_id": "INT-04", "model_name": "Model A", "score": 0, "reason": "Wrong intent"}],
        table,
    )
    assert (tmp_path / "latest.txt").exists()
    assert (run_dir / "summary.txt").exists()
    assert (run_dir / "case_results.jsonl").exists()
    assert len(read_history_csv(tmp_path)) == 1
    assert "INT-04" in format_failures([{"case_id": "INT-04", "model_name": "Model A", "score": 0, "reason": "Wrong intent"}])


def test_cli_smoke_dry_run(tmp_path: Path) -> None:
    # Override results_dir via a temp config copy
    cfg_text = Path("configs/evaluation.yaml").read_text(encoding="utf-8").replace(
        "results_dir: data/evals/results", f"results_dir: {tmp_path.as_posix()}"
    )
    tmp_cfg = tmp_path / "eval.yaml"
    tmp_cfg.write_text(cfg_text, encoding="utf-8")

    code = cli_main(["--config", str(tmp_cfg), "--smoke", "--dry-run"])
    assert code == 0
    assert (tmp_path / "latest.txt").exists()
    assert (tmp_path / "history.csv").exists()
