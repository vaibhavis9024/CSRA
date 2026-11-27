"""OpenAI-compatible Chat Completions adapter with shared async HTTP client."""

from __future__ import annotations

import json
from typing import Any

import httpx

from evals.models.base import (
    AdapterError,
    BaseModelAdapter,
    NormalizedResponse,
    TokenUsage,
    ToolCall,
)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _parse_json_if_possible(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class OpenAICompatibleAdapter(BaseModelAdapter):
    """Adapter for OpenAI-compatible chat/completions APIs."""

    provider_name = "openai_compatible"
    default_base_url = DEFAULT_OPENAI_BASE_URL

    def __init__(
        self,
        *args: Any,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client

    async def aclose(self) -> None:
        """Close the shared httpx.AsyncClient if owned by this adapter."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        full_messages: list[dict[str, Any]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": full_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if response_schema:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]
        return payload

    async def _send_request(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> NormalizedResponse:
        base = (self.base_url or self.default_base_url).rstrip("/")
        url = f"{base}/chat/completions"
        headers = self._build_headers()
        payload = self._build_payload(messages, system_prompt, tools, response_schema)

        client = self._get_client()
        resp = await client.post(url, headers=headers, json=payload)

        # Graceful fallback if a model rejects response_format=json_object
        if resp.status_code == 400 and response_schema and "response_format" in resp.text.lower():
            fallback_payload = {k: v for k, v in payload.items() if k != "response_format"}
            resp = await client.post(url, headers=headers, json=fallback_payload)

        if resp.status_code in (429, 500, 502, 503, 504):
            raise AdapterError(
                f"{self.provider_name} API transient error ({resp.status_code}): {resp.text[:300]}",
                error_type="rate_limit" if resp.status_code == 429 else "provider_outage",
                status_code=resp.status_code,
                retryable=True,
            )
        if resp.status_code >= 400:
            error_type = "auth_error" if resp.status_code in (401, 403) else "client_error"
            raise AdapterError(
                f"{self.provider_name} API error ({resp.status_code}): {resp.text[:300]}",
                error_type=error_type,
                status_code=resp.status_code,
                retryable=False,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise AdapterError(
                f"Malformed JSON response from {self.provider_name}: {exc}",
                error_type="malformed_response",
                retryable=False,
            ) from exc

        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        text = message.get("content")

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            parse_error: str | None = None
            parsed_args: dict[str, Any] = {}
            if isinstance(raw_args, dict):
                parsed_args = raw_args
            elif isinstance(raw_args, str):
                try:
                    loaded = json.loads(raw_args)
                    if isinstance(loaded, dict):
                        parsed_args = loaded
                    else:
                        parse_error = "Tool arguments JSON is not an object"
                except json.JSONDecodeError as exc:
                    parse_error = f"Invalid JSON in tool arguments: {exc}"

            tool_calls.append(
                ToolCall(
                    id=tc.get("id"),
                    name=fn.get("name", ""),
                    arguments=parsed_args,
                    raw_arguments=raw_args if isinstance(raw_args, str) else None,
                    parse_error=parse_error,
                )
            )

        usage_raw = data.get("usage") or {}
        input_tokens = usage_raw.get("prompt_tokens")
        output_tokens = usage_raw.get("completion_tokens")
        total_tokens = usage_raw.get("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        cost_usd = usage_raw.get("cost")
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
            available=input_tokens is not None and output_tokens is not None,
        )

        return NormalizedResponse(
            model_name=self.model_name,
            provider=self.provider_name,
            model_id=self.model_id,
            text=text,
            tool_calls=tool_calls,
            structured_output=_parse_json_if_possible(text) if response_schema else None,
            usage=usage,
            raw_response=data,
        )
