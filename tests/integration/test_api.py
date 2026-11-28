"""Integration tests for FastAPI Stage 0 endpoints using mocked LLM transport."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from csra.api.routes import get_llm_client
from csra.main import app
from csra.schemas.support import SupportResponse
from evals.models.cloudflare import CloudflareAdapter, build_cloudflare_base_url


def _override_adapter(handler: object) -> CloudflareAdapter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
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


def test_support_endpoint_success() -> None:
    llm_payload = {
        "primary_intent": "payment_issue",
        "secondary_intents": [],
        "order_id": None,
        "issue_summary": "The customer reports a payment deduction while the order says unpaid.",
        "extracted_entities": {"payment_issue": "payment deducted but order marked unpaid"},
        "missing_information": [],
        "needs_clarification": False,
        "confidence": 0.95,
    }
    mock_adapter = _override_adapter(
        lambda _req: httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(llm_payload)}}]})
    )
    app.dependency_overrides[get_llm_client] = lambda: mock_adapter

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/support",
            json={
                "session_id": "session_001",
                "customer_id": "customer_101",
                "message": "My payment was deducted but my order still says unpaid.",
            },
        )

    assert resp.status_code == 200
    validated = SupportResponse.model_validate(resp.json())
    assert validated.session_id == "session_001"
    assert validated.issue_category == "payment_issue"
    assert validated.processing_status == "analyzed"


def test_support_endpoint_invalid_payload_returns_422() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/support",
            json={"session_id": "s1", "customer_id": "c1", "message": ""},
        )
    assert resp.status_code == 422


def test_support_endpoint_provider_failure_returns_502() -> None:
    mock_adapter = _override_adapter(lambda _req: httpx.Response(503, text="Upstream down"))
    app.dependency_overrides[get_llm_client] = lambda: mock_adapter

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/support",
            json={"session_id": "s1", "customer_id": "c1", "message": "Where is my order?"},
        )
    assert resp.status_code == 502
