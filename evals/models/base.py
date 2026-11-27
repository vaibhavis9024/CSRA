"""Base provider adapter and normalized response schemas."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Normalized token usage reported by a provider."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    available: bool = False


class ToolCall(BaseModel):
    """Normalized tool/function call emitted by a model."""

    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None
    parse_error: str | None = None


class NormalizedResponse(BaseModel):
    """Provider-agnostic response returned by all adapters."""

    model_name: str
    provider: str
    model_id: str
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    structured_output: dict[str, Any] | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    retry_count: int = 0
    error: str | None = None
    error_type: str | None = None
    raw_response: dict[str, Any] | None = None


class AdapterError(Exception):
    """Normalized error raised by provider adapters."""

    def __init__(
        self,
        message: str,
        error_type: str = "provider_error",
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = retryable


class BaseModelAdapter(ABC):
    """Abstract base class for all LLM provider adapters."""

    provider_name: str = "base"

    def __init__(
        self,
        model_name: str,
        model_id: str,
        api_key: str | None = None,
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_id = model_id
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.base_url = base_url
        self.extra_headers = extra_headers or {}

    async def aclose(self) -> None:
        """Release any underlying HTTP client resources."""
        return None

    @abstractmethod
    async def _send_request(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> NormalizedResponse:
        """Execute a single provider API request and normalize the response."""

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> NormalizedResponse:
        """Execute a model request with bounded retries, timeout, and error normalization."""
        attempt = 0
        last_error: AdapterError | None = None

        while attempt <= self.max_retries:
            start = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self._send_request(
                        messages=messages,
                        system_prompt=system_prompt,
                        tools=tools,
                        response_schema=response_schema,
                    ),
                    timeout=self.timeout_seconds,
                )
                response.latency_ms = (time.perf_counter() - start) * 1000.0
                response.retry_count = attempt
                return response
            except TimeoutError:
                last_error = AdapterError(
                    f"Request timed out after {self.timeout_seconds}s",
                    error_type="timeout",
                    retryable=True,
                )
            except httpx.TimeoutException as exc:
                last_error = AdapterError(
                    f"HTTP timeout: {exc}",
                    error_type="timeout",
                    retryable=True,
                )
            except httpx.RequestError as exc:
                last_error = AdapterError(
                    f"Network error: {exc}",
                    error_type="network_error",
                    retryable=True,
                )
            except AdapterError as exc:
                last_error = exc
            except Exception as exc:  # noqa: BLE001
                last_error = AdapterError(
                    f"Unexpected adapter error: {exc}",
                    error_type="unexpected_error",
                    retryable=False,
                )

            if not last_error.retryable or attempt >= self.max_retries:
                break

            backoff = self.retry_backoff_seconds * (2**attempt)
            await asyncio.sleep(backoff)
            attempt += 1

        elapsed_ms = self.timeout_seconds * 1000.0 if last_error and last_error.error_type == "timeout" else 0.0
        return NormalizedResponse(
            model_name=self.model_name,
            provider=self.provider_name,
            model_id=self.model_id,
            latency_ms=elapsed_ms,
            retry_count=attempt,
            error=str(last_error) if last_error else "Unknown error",
            error_type=last_error.error_type if last_error else "unknown_error",
        )
