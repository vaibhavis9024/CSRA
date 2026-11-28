"""FastAPI routes for CSRA Stage 0 & Stage 1."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from csra.agent.graph import build_stage1_graph
from csra.llm.client import LLMServiceError, create_llm_client
from csra.schemas.support import (
    InvestigationSummary,
    IssueAnalysis,
    SupportRequest,
    SupportResponse,
)
from csra.tools.registry import ToolRegistry, get_tool_registry
from evals.models.base import BaseModelAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Support"])

ERROR_STATUS_MAP: dict[str, int] = {
    "rate_limit": 429,
    "timeout": 504,
    "config_error": 503,
    "auth_error": 502,
    "provider_outage": 502,
    "network_error": 502,
    "malformed_output": 502,
    "validation_error": 502,
    "provider_error": 502,
    "client_error": 502,
    "workflow_error": 500,
}


def get_llm_client(request: Request) -> BaseModelAdapter:
    """Resolve or lazily initialize the shared LLM client from app state."""
    client: BaseModelAdapter | None = getattr(request.app.state, "llm_client", None)
    if client is None:
        try:
            client = create_llm_client()
        except LLMServiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        request.app.state.llm_client = client
    return client


@router.post("/support", response_model=SupportResponse)
async def handle_support_request(
    payload: SupportRequest,
    llm_client: Annotated[BaseModelAdapter, Depends(get_llm_client)],
    tool_registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
) -> SupportResponse:
    """Understand and investigate an incoming customer support request."""
    start = time.perf_counter()
    logger.info("Workflow started for session=%s", payload.session_id)

    graph = build_stage1_graph(llm_client, tool_registry)
    result = await graph.ainvoke(
        {
            "session_id": payload.session_id,
            "customer_id": payload.customer_id,
            "message": payload.message,
            "order_id": payload.order_id,
            "processing_status": "started",
        }
    )

    duration_ms = (time.perf_counter() - start) * 1000.0
    if result.get("processing_status") == "failed" or result.get("issue_analysis") is None:
        err_type = result.get("error_type") or "workflow_error"
        status_code = ERROR_STATUS_MAP.get(err_type, 502)
        logger.warning(
            "Workflow failed for session=%s duration_ms=%.2f error_type=%s",
            payload.session_id,
            duration_ms,
            err_type,
        )
        raise HTTPException(
            status_code=status_code,
            detail=result.get("error") or "Failed to process support request.",
        )

    analysis: IssueAnalysis = result["issue_analysis"]
    summary: InvestigationSummary | None = result.get("investigation_summary")
    tools_executed = [r.tool_name for r in result.get("tool_results", [])]

    logger.info(
        "Workflow completed for session=%s intent=%s status=%s tool_calls=%d duration_ms=%.2f",
        payload.session_id,
        analysis.primary_intent,
        summary.investigation_status if summary else "unknown",
        len(tools_executed),
        duration_ms,
    )

    return SupportResponse(
        session_id=payload.session_id,
        issue_category=analysis.primary_intent,
        secondary_intents=analysis.secondary_intents,
        order_id=analysis.order_id,
        issue_summary=analysis.issue_summary,
        extracted_entities=analysis.extracted_entities,
        missing_information=analysis.missing_information,
        needs_clarification=analysis.needs_clarification,
        confidence=analysis.confidence,
        processing_status=result["processing_status"],
        investigation_status=summary.investigation_status if summary else "incomplete",
        findings=summary.findings if summary else [],
        evidence_summary=summary.evidence_summary if summary else "",
        unresolved_questions=summary.unresolved_questions if summary else [],
        investigation_complete=summary.investigation_complete if summary else False,
        tools_executed=tools_executed,
    )
