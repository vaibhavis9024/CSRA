"""Stage 0 LangGraph workflow: Understand Request -> Validate Structured Output."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from csra.agent.state import SupportAgentState
from csra.llm.client import LLMServiceError, analyze_support_request
from csra.schemas.support import IssueAnalysis, SupportRequest
from evals.models.base import BaseModelAdapter

logger = logging.getLogger(__name__)


def build_stage0_graph(llm_client: BaseModelAdapter) -> Any:
    """Compile the Stage 0 request-understanding LangGraph workflow."""

    async def understand_request(state: SupportAgentState) -> dict[str, Any]:
        session_id = state["session_id"]
        logger.info("Node 'understand_request' started for session=%s", session_id)
        try:
            req = SupportRequest(
                session_id=session_id,
                customer_id=state["customer_id"],
                message=state["message"],
                order_id=state.get("order_id"),
            )
            raw = await analyze_support_request(llm_client, req)
            return {"raw_analysis": raw, "processing_status": "llm_completed"}
        except LLMServiceError as exc:
            logger.warning("LLMServiceError in session=%s: %s", session_id, exc)
            return {
                "raw_analysis": None,
                "processing_status": "failed",
                "error": str(exc),
                "error_type": exc.error_type,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in understand_request for session=%s", session_id)
            return {
                "raw_analysis": None,
                "processing_status": "failed",
                "error": f"Unexpected workflow error: {exc}",
                "error_type": "workflow_error",
            }

    async def validate_output(state: SupportAgentState) -> dict[str, Any]:
        session_id = state["session_id"]
        logger.info("Node 'validate_output' started for session=%s", session_id)
        if state.get("error"):
            return {"issue_analysis": None, "processing_status": "failed"}

        raw = state.get("raw_analysis")
        if raw is None:
            return {
                "processing_status": "failed",
                "error": "No analysis produced by model.",
                "error_type": "validation_error",
            }

        try:
            analysis = IssueAnalysis.model_validate(raw)
            if analysis.order_id is None and state.get("order_id"):
                analysis = analysis.model_copy(update={"order_id": state["order_id"]})
            return {"issue_analysis": analysis, "processing_status": "analyzed"}
        except ValidationError as exc:
            logger.warning("Structured output validation failed for session=%s: %s", session_id, exc)
            return {
                "issue_analysis": None,
                "processing_status": "failed",
                "error": f"Structured output validation failed: {exc}",
                "error_type": "validation_error",
            }

    graph = StateGraph(SupportAgentState)
    graph.add_node("understand_request", understand_request)
    graph.add_node("validate_output", validate_output)

    graph.add_edge(START, "understand_request")
    graph.add_edge("understand_request", "validate_output")
    graph.add_edge("validate_output", END)

    return graph.compile()
