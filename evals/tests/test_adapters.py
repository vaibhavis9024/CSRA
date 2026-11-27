"""Tests for Cloudflare Workers AI adapter, registry, error handling, and retries."""

from __future__ import annotations

import json

import httpx
import pytest

from csra.utils.config import Settings
from evals.models.adapters.cloudflare import CloudflareAdapter, build_cloudflare_base_url
from evals.models.registry import ModelRegistry, load_evaluation_config


def test_registry_loads_cloudflare_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-test-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
    cfg = load_evaluation_config("configs/evaluation.yaml")
    reg = ModelRegistry(cfg, settings=Settings(_env_file=None))
    enabled = reg.enabled_models()
    assert len(enabled) == 5
    adapter = reg.create_adapter("@cf/meta/llama-3.3-70b-instruct-fp8-fast", model_name="Llama 3.3 70B")
    assert isinstance(adapter, CloudflareAdapter)
    assert adapter.model_id == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    assert adapter.base_url == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1"


def test_registry_missing_credentials_and_custom_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_BASE_URL", raising=False)
    cfg = load_evaluation_config("configs/evaluation.yaml")
    reg = ModelRegistry(cfg, settings=Settings(_env_file=None))

    with pytest.raises(ValueError, match="CLOUDFLARE_API_KEY"):
        reg.create_adapter("@cf/qwen/qwq-32b", require_api_key=True)

    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-test-key")
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        reg.create_adapter("@cf/qwen/qwq-32b", require_api_key=True)

    monkeypatch.setenv("CLOUDFLARE_BASE_URL", "https://custom.cf.local/v1")
    adapter = reg.create_adapter("@cf/qwen/qwq-32b", require_api_key=True)
    assert adapter.base_url == "https://custom.cf.local/v1"


@pytest.mark.asyncio
async def test_cloudflare_request_payload_and_response_parsing() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"intents": ["billing", "delivery"]}',
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_order",
                                        "arguments": json.dumps({"order_id": "ORD-1001"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = CloudflareAdapter(
            model_name="Llama 3.3 70B",
            model_id="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            api_key="cf-secret",
            base_url=build_cloudflare_base_url("acc123"),
            client=client,
        )
        resp = await adapter.generate(
            messages=[{"role": "user", "content": "Check ORD-1001"}],
            system_prompt="You are helpful.",
            tools=[{"name": "lookup_order", "description": "Lookup", "parameters": {"type": "object"}}],
            response_schema={"type": "object"},
        )

    assert len(captured) == 1
    req = captured[0]
    assert str(req.url) == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer cf-secret"
    assert resp.error is None
    assert resp.provider == "cloudflare"
    assert resp.structured_output == {"intents": ["billing", "delivery"]}
    assert resp.tool_calls[0].name == "lookup_order"
    assert resp.usage.total_tokens == 150


@pytest.mark.asyncio
async def test_cloudflare_auth_failure_not_retried() -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "Invalid API token"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = CloudflareAdapter(
            model_name="m",
            model_id="@cf/qwen/qwq-32b",
            api_key="bad-key",
            base_url=build_cloudflare_base_url("acc123"),
            max_retries=2,
            retry_backoff_seconds=0.01,
            client=client,
        )
        resp = await adapter.generate([{"role": "user", "content": "hi"}])

    assert calls == 1
    assert resp.error_type == "auth_error"


@pytest.mark.asyncio
async def test_cloudflare_rate_limit_retried_and_recovers() -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="Rate limit exceeded")
        return httpx.Response(200, json={"choices": [{"message": {"content": "recovered"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = CloudflareAdapter(
            model_name="m",
            model_id="@cf/google/gemma-4-26b-a4b-it",
            api_key="cf-test",
            base_url=build_cloudflare_base_url("acc123"),
            max_retries=2,
            retry_backoff_seconds=0.01,
            client=client,
        )
        resp = await adapter.generate([{"role": "user", "content": "hi"}])

    assert calls == 2
    assert resp.error is None
    assert resp.text == "recovered"
