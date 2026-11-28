"""Unit tests for Stage 0 Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from csra.schemas.support import IssueAnalysis, SupportRequest


def test_valid_support_request() -> None:
    req = SupportRequest(
        session_id="session_001",
        customer_id="customer_101",
        message="My payment was deducted but my order still says unpaid.",
    )
    assert req.order_id is None


def test_empty_message_rejected() -> None:
    with pytest.raises(ValidationError):
        SupportRequest(session_id="s1", customer_id="c1", message="")


def test_overlong_message_rejected() -> None:
    with pytest.raises(ValidationError):
        SupportRequest(session_id="s1", customer_id="c1", message="x" * 5001)


def test_valid_issue_analysis() -> None:
    analysis = IssueAnalysis(
        primary_intent="payment_issue",
        secondary_intents=[],
        order_id=None,
        issue_summary="Payment deducted while order remains unpaid.",
        extracted_entities={"payment_issue": "deducted but unpaid"},
        missing_information=[],
        needs_clarification=False,
        confidence=0.95,
    )
    assert analysis.primary_intent == "payment_issue"


@pytest.mark.parametrize("bad_conf", [-0.1, 1.01])
def test_invalid_confidence_rejected(bad_conf: float) -> None:
    with pytest.raises(ValidationError):
        IssueAnalysis(
            primary_intent="order_issue",
            issue_summary="Summary",
            confidence=bad_conf,
        )
