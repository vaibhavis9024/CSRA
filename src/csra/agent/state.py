"""Minimal LangGraph state schema for Stage 0."""

from __future__ import annotations

from typing import Any, TypedDict

from csra.schemas.support import IssueAnalysis


class SupportAgentState(TypedDict, total=False):
    """State passed through the Stage 0 LangGraph workflow."""

    session_id: str
    customer_id: str
    message: str
    order_id: str | None
    raw_analysis: dict[str, Any] | None
    issue_analysis: IssueAnalysis | None
    processing_status: str
    error: str | None
    error_type: str | None
