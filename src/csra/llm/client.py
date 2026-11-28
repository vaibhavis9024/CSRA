"""LLM client factory and Stage 0 / Stage 1 structured LLM calls."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from csra.schemas.support import IssueAnalysis, SupportRequest, ToolResult
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

INVESTIGATION_PLAN_PROMPT = """You are the investigation planner for CSRA (Customer Support Resolution Agent).
Your job is to decide which read-only investigation tools to call next based on the customer issue and evidence gathered so far.

Strict rules:
- Request ONLY tools from the provided available_tools list.
- Request at most 3 tools per plan.
- Do NOT repeat tools already called with the same arguments.
- If you already have enough evidence to explain the issue, set investigation_complete=true and tool_requests=[].
- If customer information is missing and cannot be looked up, list it in missing_information and set tool_requests=[].
- Respond ONLY with valid JSON matching:
  {
    "tool_requests": [
      {"tool_name": "<name>", "arguments": {"<arg>": "<val>"}, "purpose": "<why needed>"}
    ],
    "investigation_complete": <true|false>,
    "missing_information": ["<string>"],
    "investigation_reasoning_summary": "<brief summary>"
  }
"""

INVESTIGATION_SUMMARY_PROMPT = """You are the investigation summarizer for CSRA (Customer Support Resolution Agent).
Summarize ONLY factual findings directly supported by the tool results gathered so far.

Strict rules:
- State factual findings supported by tool outputs.
- Explicitly note unavailable, failed, or missing information.
- NEVER invent facts, policies, or eligibility.
- NEVER propose a resolution or claim an action (refund, cancellation, replacement) was performed.
- Use investigation_status from: "completed", "incomplete", "needs_clarification", "failed".
- Respond ONLY with valid JSON matching:
  {
    "investigation_status": "<completed|incomplete|needs_clarification|failed>",
    "findings": ["<factual finding>"],
    "evidence_summary": "<concise grounded summary>",
    "unresolved_questions": ["<string>"],
    "customer_information_required": ["<string>"],
    "investigation_complete": <true|false>
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


async def _call_json_llm(
    client: BaseModelAdapter,
    session_id: str,
    step_name: str,
    system_prompt: str,
    user_content: str,
) -> dict[str, Any]:
    """Shared helper to call the LLM and parse a JSON object response."""
    logger.info("LLM %s started for session=%s model=%s", step_name, session_id, client.model_id)
    response = await client.generate(
        messages=[{"role": "user", "content": user_content}],
        system_prompt=system_prompt,
        response_schema={"type": "object"},
    )
    logger.info(
        "LLM %s completed for session=%s latency_ms=%.2f error_type=%s",
        step_name,
        session_id,
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


async def analyze_support_request(
    client: BaseModelAdapter,
    request: SupportRequest,
) -> dict[str, Any]:
    """Call the LLM to produce raw structured issue analysis."""
    user_content = request.message
    if request.order_id:
        user_content = f"[Order ID provided in context: {request.order_id}]\n{request.message}"
    return await _call_json_llm(
        client=client,
        session_id=request.session_id,
        step_name="understand_request",
        system_prompt=STAGE0_SYSTEM_PROMPT,
        user_content=user_content,
    )


async def plan_investigation(
    client: BaseModelAdapter,
    session_id: str,
    customer_id: str,
    message: str,
    analysis: IssueAnalysis,
    available_tools: list[dict[str, str]],
    tool_results: list[ToolResult],
    iteration: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Call the LLM to plan the next set of read-only investigation tool calls."""
    evidence = [r.model_dump() for r in tool_results]
    user_content = json.dumps(
        {
            "customer_id": customer_id,
            "customer_message": message,
            "issue_analysis": analysis.model_dump(),
            "iteration": iteration,
            "max_iterations": max_iterations,
            "available_tools": available_tools,
            "evidence_so_far": evidence,
        }
    )
    return await _call_json_llm(
        client=client,
        session_id=session_id,
        step_name="plan_investigation",
        system_prompt=INVESTIGATION_PLAN_PROMPT,
        user_content=user_content,
    )


async def summarize_investigation(
    client: BaseModelAdapter,
    session_id: str,
    message: str,
    analysis: IssueAnalysis,
    tool_results: list[ToolResult],
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    """Call the LLM to produce an evidence-grounded investigation summary."""
    user_content = json.dumps(
        {
            "customer_message": message,
            "issue_analysis": analysis.model_dump(),
            "tool_results": [r.model_dump() for r in tool_results],
            "incomplete_reason": incomplete_reason,
        }
    )
    return await _call_json_llm(
        client=client,
        session_id=session_id,
        step_name="summarize_investigation",
        system_prompt=INVESTIGATION_SUMMARY_PROMPT,
        user_content=user_content,
    )
