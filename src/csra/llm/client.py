"""LLM client factory and Stage 0 request-understanding call."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from csra.schemas.support import SupportRequest
from csra.utils.config import Settings, get_settings
from evals.models.base import BaseModelAdapter
from evals.models.cloudflare import CloudflareAdapter, build_cloudflare_base_url

logger = logging.getLogger(__name__)

STAGE0_SYSTEM_PROMPT = """You are the request-understanding module of CSRA (Customer Support Resolution Agent).
Your ONLY job in this stage is to understand the customer's message and return structured JSON.

Supported primary_intent categories (use snake_case):
- payment_issue
- order_issue
- delivery_issue
- cancellation_request
- refund_request
- return_request
- replacement_request
- product_issue
- account_issue
- troubleshooting
- information_request
- complaint
- unknown_or_ambiguous

Strict rules:
- Identify primary_intent and any secondary_intents if multiple issues appear.
- Extract only explicitly stated entities into extracted_entities (string key-value pairs).
- If the request is vague (e.g., "Help me with my order"), set needs_clarification=true and list what is missing in missing_information.
- NEVER invent customer, order, or payment details.
- NEVER claim you investigated an order, called a tool, or executed an action.
- Respond ONLY with a valid JSON object matching:
  {
    "primary_intent": "<string>",
    "secondary_intents": ["<string>"],
    "order_id": "<string or null>",
    "issue_summary": "<concise 1-sentence summary>",
    "extracted_entities": {"<key>": "<value>"},
    "missing_information": ["<string>"],
    "needs_clarification": <true|false>,
    "confidence": <float 0.0-1.0>
  }
"""


class LLMServiceError(Exception):
    """Error raised when the LLM provider fails or returns invalid output."""

    def __init__(self, message: str, error_type: str = "provider_error") -> None:
        super().__init__(message)
        self.error_type = error_type


def create_llm_client(settings: Settings | None = None) -> CloudflareAdapter:
    """Create a CloudflareAdapter from centralized application settings."""
    cfg = settings or get_settings()
    api_key = (
        os.getenv("CLOUDFLARE_API_KEY")
        or os.getenv("CLOUDFLARE_API_TOKEN")
        or (cfg.cloudflare_api_key.get_secret_value() if cfg.cloudflare_api_key else None)
    )
    if not api_key:
        raise LLMServiceError(
            "Missing Cloudflare API key (set CLOUDFLARE_API_KEY).",
            error_type="config_error",
        )

    explicit_url = os.getenv("CLOUDFLARE_BASE_URL") or cfg.cloudflare_base_url
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID") or cfg.cloudflare_account_id
    if explicit_url:
        base_url = explicit_url.rstrip("/")
    elif account_id:
        base_url = build_cloudflare_base_url(account_id)
    else:
        raise LLMServiceError(
            "Missing Cloudflare Account ID (set CLOUDFLARE_ACCOUNT_ID).",
            error_type="config_error",
        )

    return CloudflareAdapter(
        model_name=cfg.llm_model_name,
        model_id=cfg.llm_model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_output_tokens=1024,
        timeout_seconds=60.0,
        max_retries=2,
    )


async def analyze_support_request(
    client: BaseModelAdapter,
    request: SupportRequest,
) -> dict[str, Any]:
    """Call the LLM to produce raw structured issue analysis."""
    user_content = request.message
    if request.order_id:
        user_content = f"[Order ID provided in context: {request.order_id}]\n{request.message}"

    logger.info("LLM request started for session=%s model=%s", request.session_id, client.model_id)
    response = await client.generate(
        messages=[{"role": "user", "content": user_content}],
        system_prompt=STAGE0_SYSTEM_PROMPT,
        response_schema={"type": "object"},
    )
    logger.info(
        "LLM request completed for session=%s latency_ms=%.2f error_type=%s",
        request.session_id,
        response.latency_ms,
        response.error_type,
    )

    if response.error is not None:
        raise LLMServiceError(
            response.error,
            error_type=response.error_type or "provider_error",
        )

    if response.structured_output is not None:
        return response.structured_output

    if response.text:
        try:
            parsed = json.loads(response.text.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise LLMServiceError(
                f"Malformed JSON from model: {exc}",
                error_type="malformed_output",
            ) from exc

    raise LLMServiceError("Empty or non-JSON response from model.", error_type="malformed_output")
