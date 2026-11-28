"""LangGraph state schema for Stage 0 and Stage 1."""

from __future__ import annotations

from typing import Any, TypedDict

from csra.schemas.support import (
    InvestigationPlan,
    InvestigationSummary,
    IssueAnalysis,
    ToolResult,
)


class SupportAgentState(TypedDict, total=False):
    """State passed through the CSRA LangGraph workflow."""

    session_id: str
    customer_id: str
    message: str
    order_id: str | None
    raw_analysis: dict[str, Any] | None
    issue_analysis: IssueAnalysis | None

    # Stage 1 investigation state
    investigation_plan: InvestigationPlan | None
    tool_results: list[ToolResult]
    executed_calls: list[str]
    iteration_count: int
    tool_call_count: int
    investigation_status: str
    missing_information: list[str]
    investigation_summary: InvestigationSummary | None
    limit_reached: bool
    incomplete_reason: str | None

    processing_status: str
    error: str | None
    error_type: str | None
