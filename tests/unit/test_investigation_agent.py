"""Unit tests for Stage 1 LangGraph bounded investigation workflow."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from csra.agent.graph import build_stage1_graph
from csra.tools.registry import ToolRegistry
from evals.models.cloudflare import CloudflareAdapter, build_cloudflare_base_url


def _queued_adapter(responses: list[dict[str, Any]]) -> CloudflareAdapter:
    idx = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal idx
        payload = responses[min(idx, len(responses) - 1)]
        idx += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return CloudflareAdapter(
        model_name="mock-mistral",
        model_id="@cf/mistralai/mistral-small-3.1-24b-instruct",
        api_key="test-key",
        base_url=build_cloudflare_base_url("test-acc"),
        max_retries=0,
        client=client,
    )


@pytest.mark.asyncio
async def test_payment_issue_multi_step_investigation() -> None:
    adapter = _queued_adapter([
        {
            "primary_intent": "payment_issue",
            "secondary_intents": [],
            "order_id": None,
            "issue_summary": "Customer reports payment deducted while order says unpaid.",
            "extracted_entities": {},
            "missing_information": [],
            "needs_clarification": False,
            "confidence": 0.95,
        },
        {
            "tool_requests": [
                {"tool_name": "get_customer_orders", "arguments": {}, "purpose": "Find customer orders"}
            ],
            "investigation_complete": False,
            "missing_information": [],
            "investigation_reasoning_summary": "Look up customer orders first.",
        },
        {
            "tool_requests": [
                {"tool_name": "get_payment_status", "arguments": {"order_id": "ORD-1001"}, "purpose": "Verify payment"}
            ],
            "investigation_complete": False,
            "missing_information": [],
            "investigation_reasoning_summary": "Check payment for unpaid order ORD-1001.",
        },
        {
            "tool_requests": [],
            "investigation_complete": True,
            "missing_information": [],
            "investigation_reasoning_summary": "Evidence gathered.",
        },
        {
            "investigation_status": "completed",
            "findings": ["Order ORD-1001 is unpaid.", "Payment PAY-5001 shows status deducted."],
            "evidence_summary": "Order ORD-1001 is unpaid while payment PAY-5001 is deducted.",
            "unresolved_questions": [],
            "customer_information_required": [],
            "investigation_complete": True,
        },
    ])
    graph = build_stage1_graph(adapter, ToolRegistry())
    result = await graph.ainvoke({
        "session_id": "s1",
        "customer_id": "customer_101",
        "message": "My payment was deducted but my order still says unpaid.",
    })
    await adapter.aclose()

    assert result["processing_status"] == "investigated"
    assert result["investigation_status"] == "completed"
    assert [r.tool_name for r in result["tool_results"]] == ["get_customer_orders", "get_payment_status"]
    assert result["investigation_summary"].investigation_complete is True


@pytest.mark.asyncio
async def test_delivery_issue_investigation() -> None:
    adapter = _queued_adapter([
        {
            "primary_intent": "delivery_issue",
            "order_id": "ORD-1003",
            "issue_summary": "Customer asks about delayed shipment.",
            "needs_clarification": False,
            "confidence": 0.92,
        },
        {
            "tool_requests": [
                {"tool_name": "get_delivery_status", "arguments": {"order_id": "ORD-1003"}, "purpose": "Check delivery"}
            ],
            "investigation_complete": True,
            "missing_information": [],
            "investigation_reasoning_summary": "Check delivery status.",
        },
        {
            "investigation_status": "completed",
            "findings": ["Shipment TRK-3003 is delayed due to weather_delay."],
            "evidence_summary": "Order ORD-1003 is delayed due to weather.",
            "unresolved_questions": [],
            "customer_information_required": [],
            "investigation_complete": True,
        },
    ])
    graph = build_stage1_graph(adapter, ToolRegistry())
    result = await graph.ainvoke({
        "session_id": "s2",
        "customer_id": "customer_102",
        "message": "Where is ORD-1003?",
        "order_id": "ORD-1003",
    })
    await adapter.aclose()

    assert result["investigation_status"] == "completed"
    assert result["tool_results"][0].data["delivery_exception"] == "weather_delay"


@pytest.mark.asyncio
async def test_clarification_skips_tools() -> None:
    adapter = _queued_adapter([
        {
            "primary_intent": "unknown_or_ambiguous",
            "issue_summary": "Vague request.",
            "missing_information": ["specific_issue"],
            "needs_clarification": True,
            "confidence": 0.8,
        }
    ])
    graph = build_stage1_graph(adapter, ToolRegistry())
    result = await graph.ainvoke({
        "session_id": "s3",
        "customer_id": "customer_101",
        "message": "Help me with my order.",
    })
    await adapter.aclose()

    assert result["investigation_status"] == "needs_clarification"
    assert result.get("tool_results", []) == []


@pytest.mark.asyncio
async def test_iteration_limit_and_duplicate_call_prevention() -> None:
    loop_plan = {
        "tool_requests": [
            {"tool_name": "get_customer_orders", "arguments": {}, "purpose": "Check orders"}
        ],
        "investigation_complete": False,
        "missing_information": [],
        "investigation_reasoning_summary": "Keep checking.",
    }
    adapter = _queued_adapter([
        {
            "primary_intent": "order_issue",
            "issue_summary": "Order check",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        loop_plan,
        loop_plan,
        loop_plan,
        {
            "investigation_status": "completed",
            "findings": ["Checked orders."],
            "evidence_summary": "Partial evidence.",
            "unresolved_questions": [],
            "customer_information_required": [],
            "investigation_complete": True,
        },
    ])
    graph = build_stage1_graph(adapter, ToolRegistry(), max_iterations=2, max_tool_calls=8)
    result = await graph.ainvoke({
        "session_id": "s4",
        "customer_id": "customer_101",
        "message": "Check my orders.",
    })
    await adapter.aclose()

    # Duplicate get_customer_orders was skipped on iteration 2; bounded at 2 iterations
    assert len(result["tool_results"]) == 1
    assert result["iteration_count"] == 2


@pytest.mark.asyncio
async def test_max_tool_calls_limit_enforced() -> None:
    adapter = _queued_adapter([
        {
            "primary_intent": "order_issue",
            "issue_summary": "Check everything",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        {
            "tool_requests": [
                {"tool_name": "get_customer_profile", "arguments": {}, "purpose": "Profile"},
                {"tool_name": "get_customer_orders", "arguments": {}, "purpose": "Orders"},
                {"tool_name": "get_order_details", "arguments": {"order_id": "ORD-1001"}, "purpose": "Details"},
            ],
            "investigation_complete": False,
            "missing_information": [],
            "investigation_reasoning_summary": "Batch 1",
        },
        {
            "investigation_status": "completed",
            "findings": ["Partial"],
            "evidence_summary": "Stopped at tool limit.",
            "unresolved_questions": [],
            "customer_information_required": [],
            "investigation_complete": True,
        },
    ])
    graph = build_stage1_graph(adapter, ToolRegistry(), max_iterations=5, max_tool_calls=2)
    result = await graph.ainvoke({
        "session_id": "s5",
        "customer_id": "customer_101",
        "message": "Check my account.",
    })
    await adapter.aclose()

    assert result["tool_call_count"] == 2
    assert result["limit_reached"] is True
    assert result["investigation_status"] == "incomplete"
    assert result["investigation_summary"].investigation_complete is False
