"""Pydantic schemas for Stage 0 CSRA request understanding."""

from __future__ import annotations

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
