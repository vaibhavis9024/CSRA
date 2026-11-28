"""Pydantic schemas for Stage 0 understanding and Stage 1 investigation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SupportRequest(BaseModel):
    """Incoming customer support API request."""

    session_id: str = Field(..., min_length=1)
    customer_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=5000)
    order_id: str | None = None


class IssueAnalysis(BaseModel):
    """Structured LLM output for Stage 0 request understanding."""

    primary_intent: str = Field(..., min_length=1)
    secondary_intents: list[str] = Field(default_factory=list)
    order_id: str | None = None
    issue_summary: str = Field(..., min_length=1)
    extracted_entities: dict[str, str] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    confidence: float = Field(..., ge=0.0, le=1.0)


class ToolRequest(BaseModel):
    """A single read-only tool call requested by the investigation planner."""

    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(..., min_length=1)


class InvestigationPlan(BaseModel):
    """Structured LLM output for planning the next investigation step."""

    tool_requests: list[ToolRequest] = Field(default_factory=list)
    investigation_complete: bool = False
    missing_information: list[str] = Field(default_factory=list)
    investigation_reasoning_summary: str = Field(..., min_length=1)


class ToolResult(BaseModel):
    """Structured result returned by the controlled tool registry."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool
    error_type: Literal[
        "not_found",
        "access_denied",
        "invalid_arguments",
        "unknown_tool",
        "execution_error",
    ] | None = None
    data: dict[str, Any] | None = None
    error: str | None = None


class InvestigationSummary(BaseModel):
    """Evidence-grounded investigation summary produced at the end of Stage 1."""

    investigation_status: Literal["completed", "incomplete", "needs_clarification", "failed"]
    findings: list[str] = Field(default_factory=list)
    evidence_summary: str = Field(..., min_length=1)
    unresolved_questions: list[str] = Field(default_factory=list)
    customer_information_required: list[str] = Field(default_factory=list)
    investigation_complete: bool = False


class SupportResponse(BaseModel):
    """API response returned by POST /api/v1/support."""

    session_id: str
    issue_category: str
    secondary_intents: list[str] = Field(default_factory=list)
    order_id: str | None = None
    issue_summary: str
    extracted_entities: dict[str, str] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_status: str
    investigation_status: str = "completed"
    findings: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)
    investigation_complete: bool = False
    tools_executed: list[str] = Field(default_factory=list)
