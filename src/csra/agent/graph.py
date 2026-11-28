"""LangGraph workflows for Stage 0 (understanding) and Stage 1 (investigation)."""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from csra.agent.state import SupportAgentState
from csra.llm.client import (
    LLMServiceError,
    analyze_support_request,
    plan_investigation,
    summarize_investigation,
)
from csra.schemas.support import (
    InvestigationPlan,
    InvestigationSummary,
    IssueAnalysis,
    SupportRequest,
    ToolResult,
)
from csra.tools.registry import ToolRegistry
from evals.models.base import BaseModelAdapter

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_TOOL_CALLS = 8
DEFAULT_MAX_TOOLS_PER_PLAN = 3


def _call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"


def _build_understanding_nodes(llm_client: BaseModelAdapter) -> tuple[Any, Any]:
    """Create shared Stage 0 understanding and validation nodes."""

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
                "issue_analysis": None,
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

    return understand_request, validate_output


def build_stage0_graph(llm_client: BaseModelAdapter) -> Any:
    """Compile the Stage 0 request-understanding LangGraph workflow."""
    understand_request, validate_output = _build_understanding_nodes(llm_client)

    graph = StateGraph(SupportAgentState)
    graph.add_node("understand_request", understand_request)
    graph.add_node("validate_output", validate_output)

    graph.add_edge(START, "understand_request")
    graph.add_edge("understand_request", "validate_output")
    graph.add_edge("validate_output", END)

    return graph.compile()


def build_stage1_graph(
    llm_client: BaseModelAdapter,
    tool_registry: ToolRegistry,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_tools_per_plan: int = DEFAULT_MAX_TOOLS_PER_PLAN,
) -> Any:
    """Compile the Stage 1 bounded investigation LangGraph workflow."""
    understand_request, validate_output = _build_understanding_nodes(llm_client)

    def route_after_understanding(state: SupportAgentState) -> str:
        if state.get("error") or state.get("issue_analysis") is None:
            return "failed"
        analysis: IssueAnalysis = state["issue_analysis"]
        if analysis.needs_clarification:
            return "needs_clarification"
        return "investigate"

    async def plan_next_step(state: SupportAgentState) -> dict[str, Any]:
        session_id = state["session_id"]
        iteration = state.get("iteration_count", 0) + 1
        logger.info("Investigation iteration=%d for session=%s", iteration, session_id)

        if iteration > max_iterations:
            logger.warning("Max iterations (%d) reached for session=%s", max_iterations, session_id)
            return {
                "limit_reached": True,
                "incomplete_reason": f"Reached maximum investigation iterations ({max_iterations}).",
                "investigation_plan": InvestigationPlan(
                    tool_requests=[],
                    investigation_complete=False,
                    missing_information=[],
                    investigation_reasoning_summary="Stopped at maximum iteration limit.",
                ),
            }

        try:
            raw_plan = await plan_investigation(
                client=llm_client,
                session_id=session_id,
                customer_id=state["customer_id"],
                message=state["message"],
                analysis=state["issue_analysis"],  # type: ignore[arg-type]
                available_tools=tool_registry.describe_tools(),
                tool_results=state.get("tool_results", []),
                iteration=iteration,
                max_iterations=max_iterations,
            )
            plan = InvestigationPlan.model_validate(raw_plan)
            return {
                "iteration_count": iteration,
                "investigation_plan": plan,
                "missing_information": plan.missing_information,
            }
        except (LLMServiceError, ValidationError) as exc:
            logger.warning("Investigation planning failed for session=%s: %s", session_id, exc)
            return {
                "iteration_count": iteration,
                "incomplete_reason": f"Investigation planning error: {exc}",
                "investigation_plan": None,
                "error_type": getattr(exc, "error_type", "validation_error"),
            }

    async def execute_tools(state: SupportAgentState) -> dict[str, Any]:
        session_id = state["session_id"]
        plan = state.get("investigation_plan")
        if plan is None or state.get("limit_reached") or not plan.tool_requests:
            return {}

        results: list[ToolResult] = list(state.get("tool_results", []))
        executed_sigs: list[str] = list(state.get("executed_calls", []))
        call_count = state.get("tool_call_count", 0)
        limit_reached = False
        incomplete_reason = state.get("incomplete_reason")

        for req in plan.tool_requests[:max_tools_per_plan]:
            if call_count >= max_tool_calls:
                limit_reached = True
                incomplete_reason = f"Reached maximum tool calls ({max_tool_calls})."
                logger.warning("Max tool calls (%d) reached for session=%s", max_tool_calls, session_id)
                break

            sig = _call_signature(req.tool_name, req.arguments)
            if sig in executed_sigs:
                logger.info("Skipping duplicate tool call %s in session=%s", sig, session_id)
                continue

            res = tool_registry.execute(
                tool_name=req.tool_name,
                arguments=req.arguments,
                request_customer_id=state["customer_id"],
            )
            results.append(res)
            executed_sigs.append(sig)
            call_count += 1

        return {
            "tool_results": results,
            "executed_calls": executed_sigs,
            "tool_call_count": call_count,
            "limit_reached": limit_reached,
            "incomplete_reason": incomplete_reason,
        }

    def check_evidence(state: SupportAgentState) -> str:
        plan = state.get("investigation_plan")
        if plan is None or state.get("limit_reached"):
            return "summarize"
        if plan.investigation_complete or plan.missing_information or not plan.tool_requests:
            return "summarize"
        if state.get("iteration_count", 0) >= max_iterations:
            return "summarize"
        if state.get("tool_call_count", 0) >= max_tool_calls:
            return "summarize"
        return "continue"

    async def generate_summary(state: SupportAgentState) -> dict[str, Any]:
        session_id = state["session_id"]
        analysis: IssueAnalysis = state["issue_analysis"]  # type: ignore[assignment]
        tool_results = state.get("tool_results", [])
        incomplete_reason = state.get("incomplete_reason")

        if analysis.needs_clarification:
            summary = InvestigationSummary(
                investigation_status="needs_clarification",
                findings=[],
                evidence_summary="Customer clarification is required before investigation can proceed.",
                unresolved_questions=analysis.missing_information,
                customer_information_required=analysis.missing_information,
                investigation_complete=False,
            )
            return {
                "investigation_summary": summary,
                "investigation_status": summary.investigation_status,
                "processing_status": "investigated",
            }

        try:
            raw_sum = await summarize_investigation(
                client=llm_client,
                session_id=session_id,
                message=state["message"],
                analysis=analysis,
                tool_results=tool_results,
                incomplete_reason=incomplete_reason,
            )
            summary = InvestigationSummary.model_validate(raw_sum)
            if incomplete_reason or state.get("limit_reached"):
                summary = summary.model_copy(
                    update={"investigation_status": "incomplete", "investigation_complete": False}
                )
            return {
                "investigation_summary": summary,
                "investigation_status": summary.investigation_status,
                "processing_status": "investigated",
            }
        except (LLMServiceError, ValidationError) as exc:
            logger.warning("Summarization failed for session=%s: %s", session_id, exc)
            # Fall back to factual tool findings without inventing anything
            findings = [
                f"{r.tool_name}: {json.dumps(r.data)}" if r.success else f"{r.tool_name} failed: {r.error}"
                for r in tool_results
            ]
            summary = InvestigationSummary(
                investigation_status="incomplete",
                findings=findings,
                evidence_summary=incomplete_reason or "Investigation stopped before summary generation.",
                unresolved_questions=state.get("missing_information", []),
                customer_information_required=state.get("missing_information", []),
                investigation_complete=False,
            )
            return {
                "investigation_summary": summary,
                "investigation_status": "incomplete",
                "processing_status": "investigated",
            }

    graph = StateGraph(SupportAgentState)
    graph.add_node("understand_request", understand_request)
    graph.add_node("validate_output", validate_output)
    graph.add_node("plan_investigation", plan_next_step)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("generate_summary", generate_summary)

    graph.add_edge(START, "understand_request")
    graph.add_edge("understand_request", "validate_output")
    graph.add_conditional_edges(
        "validate_output",
        route_after_understanding,
        {
            "failed": END,
            "needs_clarification": "generate_summary",
            "investigate": "plan_investigation",
        },
    )
    graph.add_edge("plan_investigation", "execute_tools")
    graph.add_conditional_edges(
        "execute_tools",
        check_evidence,
        {
            "continue": "plan_investigation",
            "summarize": "generate_summary",
        },
    )
    graph.add_edge("generate_summary", END)

    return graph.compile()
