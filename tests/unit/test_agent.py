"""Unit tests for Stage 0 LangGraph workflow with mocked LLM responses."""

from __future__ import annotations

import json

import httpx
import pytest

from csra.agent.graph import build_stage0_graph
from evals.models.cloudflare import CloudflareAdapter, build_cloudflare_base_url


def _make_mock_adapter(handler: object) -> CloudflareAdapter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return CloudflareAdapter(
        model_name="mock-mistral",
        model_id="@cf/mistralai/mistral-small-3.1-24b-instruct",
        api_key="test-key",
        base_url=build_cloudflare_base_url("test-acc"),
        max_retries=0,
        client=client,
    )


@pytest.mark.asyncio
async def test_agent_payment_intent_and_entities() -> None:
    payload = {
        "primary_intent": "payment_issue",
        "secondary_intents": [],
        "order_id": None,
        "issue_summary": "Customer reports payment deducted while order is marked unpaid.",
        "extracted_entities": {"payment_issue": "payment deducted but order marked unpaid"},
        "missing_information": [],
        "needs_clarification": False,
        "confidence": 0.95,
    }
    adapter = _make_mock_adapter(
        lambda _req: httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
    )
    graph = build_stage0_graph(adapter)
    result = await graph.ainvoke(
        {
            "session_id": "session_001",
            "customer_id": "customer_101",
            "message": "My payment was deducted but my order still says unpaid.",
        }
    )
    await adapter.aclose()

    assert result["processing_status"] == "analyzed"
    analysis = result["issue_analysis"]
    assert analysis.primary_intent == "payment_issue"
    assert analysis.needs_clarification is False
    assert analysis.confidence == 0.95


@pytest.mark.asyncio
async def test_agent_vague_request_needs_clarification() -> None:
    payload = {
        "primary_intent": "unknown_or_ambiguous",
        "secondary_intents": [],
        "order_id": None,
        "issue_summary": "Customer asked for help with an order without specifying the issue or order ID.",
        "extracted_entities": {},
        "missing_information": ["order_id", "specific_issue"],
        "needs_clarification": True,
        "confidence": 0.85,
    }
    adapter = _make_mock_adapter(
        lambda _req: httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
    )
    graph = build_stage0_graph(adapter)
    result = await graph.ainvoke(
        {"session_id": "s2", "customer_id": "c2", "message": "Help me with my order."}
    )
    await adapter.aclose()

    assert result["processing_status"] == "analyzed"
    assert result["issue_analysis"].needs_clarification is True
    assert "order_id" in result["issue_analysis"].missing_information


@pytest.mark.asyncio
async def test_agent_handles_malformed_json() -> None:
    adapter = _make_mock_adapter(
        lambda _req: httpx.Response(200, json={"choices": [{"message": {"content": "not valid json"}}]})
    )
    graph = build_stage0_graph(adapter)
    result = await graph.ainvoke({"session_id": "s3", "customer_id": "c3", "message": "Hi"})
    await adapter.aclose()

    assert result["processing_status"] == "failed"
    assert result["error_type"] == "malformed_output"
    assert result["issue_analysis"] is None


@pytest.mark.asyncio
async def test_agent_handles_validation_failure() -> None:
    # Missing required issue_summary and confidence out of bounds
    bad_payload = {"primary_intent": "order_issue", "confidence": 2.5}
    adapter = _make_mock_adapter(
        lambda _req: httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(bad_payload)}}]})
    )
    graph = build_stage0_graph(adapter)
    result = await graph.ainvoke({"session_id": "s4", "customer_id": "c4", "message": "Check order"})
    await adapter.aclose()

    assert result["processing_status"] == "failed"
    assert result["error_type"] == "validation_error"


@pytest.mark.asyncio
async def test_agent_handles_provider_failure() -> None:
    adapter = _make_mock_adapter(lambda _req: httpx.Response(503, text="Service unavailable"))
    graph = build_stage0_graph(adapter)
    result = await graph.ainvoke({"session_id": "s5", "customer_id": "c5", "message": "Check order"})
    await adapter.aclose()

    assert result["processing_status"] == "failed"
    assert result["error_type"] == "provider_outage"
