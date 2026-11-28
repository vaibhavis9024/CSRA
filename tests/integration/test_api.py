"""Integration tests for FastAPI Stage 1 endpoints using mocked LLM transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from csra.api.routes import get_llm_client
from csra.main import app
from csra.schemas.support import SupportResponse
from evals.models.cloudflare import CloudflareAdapter, build_cloudflare_base_url


def _queued_override_adapter(responses: list[dict[str, Any]]) -> CloudflareAdapter:
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


@pytest.fixture(autouse=True)
def _clear_overrides() -> object:
    yield
    app.dependency_overrides.clear()


def test_root_and_health_endpoints() -> None:
    with TestClient(app) as client:
        r1 = client.get("/")
        r2 = client.get("/health")
    assert r1.status_code == 200
    assert r1.json()["status"] == "running"
    assert r2.status_code == 200
    assert r2.json()["status"] == "healthy"


def test_support_endpoint_investigates_and_returns_findings() -> None:
    mock_adapter = _queued_override_adapter([
        {
            "primary_intent": "payment_issue",
            "secondary_intents": [],
            "order_id": "ORD-1001",
            "issue_summary": "The customer reports a payment deduction while the order says unpaid.",
            "extracted_entities": {"order_id": "ORD-1001"},
            "missing_information": [],
            "needs_clarification": False,
            "confidence": 0.95,
        },
        {
            "tool_requests": [
                {"tool_name": "get_order_details", "arguments": {"order_id": "ORD-1001"}, "purpose": "Check order"},
                {"tool_name": "get_payment_status", "arguments": {"order_id": "ORD-1001"}, "purpose": "Check payment"},
            ],
            "investigation_complete": True,
            "missing_information": [],
            "investigation_reasoning_summary": "Check order and payment status.",
        },
        {
            "investigation_status": "completed",
            "findings": ["Order ORD-1001 is unpaid.", "Payment PAY-5001 is deducted."],
            "evidence_summary": "Order and payment statuses are inconsistent.",
            "unresolved_questions": [],
            "customer_information_required": [],
            "investigation_complete": True,
        },
    ])
    app.dependency_overrides[get_llm_client] = lambda: mock_adapter

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/support",
            json={
                "session_id": "session_001",
                "customer_id": "customer_101",
                "message": "My payment was deducted for ORD-1001 but it still says unpaid.",
            },
        )

    assert resp.status_code == 200
    validated = SupportResponse.model_validate(resp.json())
    assert validated.processing_status == "investigated"
    assert validated.investigation_status == "completed"
    assert validated.investigation_complete is True
    assert validated.tools_executed == ["get_order_details", "get_payment_status"]
    assert len(validated.findings) == 2


def test_support_endpoint_unauthorized_order_handled_safely() -> None:
    mock_adapter = _queued_override_adapter([
        {
            "primary_intent": "order_issue",
            "order_id": "ORD-9999",
            "issue_summary": "Customer asked about ORD-9999.",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        {
            "tool_requests": [
                {"tool_name": "get_order_details", "arguments": {"order_id": "ORD-9999"}, "purpose": "Lookup"}
            ],
            "investigation_complete": True,
            "missing_information": [],
            "investigation_reasoning_summary": "Lookup ORD-9999.",
        },
        {
            "investigation_status": "incomplete",
            "findings": ["Order ORD-9999 could not be accessed for this customer."],
            "evidence_summary": "Order access was denied.",
            "unresolved_questions": ["Verify correct order ID."],
            "customer_information_required": ["order_id"],
            "investigation_complete": False,
        },
    ])
    app.dependency_overrides[get_llm_client] = lambda: mock_adapter

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/support",
            json={
                "session_id": "s_unauth",
                "customer_id": "customer_101",
                "message": "Check order ORD-9999.",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["investigation_status"] == "incomplete"
    assert "customer_999" not in json.dumps(data)


def test_support_endpoint_invalid_payload_returns_422() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/support",
            json={"session_id": "s1", "customer_id": "c1", "message": ""},
        )
    assert resp.status_code == 422


def test_support_endpoint_provider_failure_returns_502() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(503, text="Upstream down"))
    )
    bad_adapter = CloudflareAdapter(
        model_name="mock-mistral",
        model_id="@cf/mistralai/mistral-small-3.1-24b-instruct",
        api_key="test-key",
        base_url=build_cloudflare_base_url("test-acc"),
        max_retries=0,
        client=client,
    )
    app.dependency_overrides[get_llm_client] = lambda: bad_adapter

    with TestClient(app) as test_client:
        resp = test_client.post(
            "/api/v1/support",
            json={"session_id": "s1", "customer_id": "c1", "message": "Where is my order?"},
        )
    assert resp.status_code == 502
