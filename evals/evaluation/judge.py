"""Structured 0-2 LLM judge for subjective capabilities."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from evals.models.adapters.base import BaseModelAdapter


class JudgeResult(BaseModel):
    """Structured output returned by the LLM judge."""

    score: int = Field(..., ge=0, le=2)
    reason: str


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator for a customer support AI.\n"
    "Score the assistant's response strictly on a 0-2 scale:\n"
    "- 0 = Fail: incorrect, unsafe, or failed the requirement.\n"
    "- 1 = Partial: some correct elements, but notable omissions or flaws.\n"
    "- 2 = Pass: fully meets the expected behavior and scoring criteria.\n\n"
    'Respond ONLY with JSON: {"score": <0|1|2>, "reason": "<brief 1-sentence reason>"}'
)


async def judge_response(
    judge_adapter: BaseModelAdapter | None,
    evaluated_model_id: str,
    case: dict[str, Any],
    actual_text: str | None,
) -> dict[str, Any]:
    """Evaluate a response using the LLM judge, handling errors and self-judge prevention."""
    if judge_adapter is None:
        return {"score": None, "judged_by_llm": False, "reason": "LLM judge disabled."}

    if judge_adapter.model_id == evaluated_model_id:
        return {
            "score": None,
            "judged_by_llm": False,
            "reason": "Skipped: judge model matches evaluated model.",
        }

    if not actual_text:
        return {"score": 0, "judged_by_llm": True, "reason": "Empty model response."}

    prompt = (
        f"Capability: {case['capability']}\n"
        f"Conversation: {json.dumps(case.get('messages', []))}\n"
        f"Context: {case.get('context') or 'None'}\n"
        f"Expected behavior: {case.get('expected_behavior', '')}\n"
        f"Scoring criteria: {case.get('scoring_criteria', '')}\n"
        f"Assistant response:\n{actual_text}"
    )
    resp = await judge_adapter.generate(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=JUDGE_SYSTEM_PROMPT,
        response_schema={"type": "object"},
    )
    if resp.error or not resp.text:
        return {"score": None, "judged_by_llm": True, "reason": f"Judge error: {resp.error or 'empty'}"}

    try:
        parsed = resp.structured_output or json.loads(resp.text)
        validated = JudgeResult.model_validate(parsed)
        return {"score": validated.score, "judged_by_llm": True, "reason": validated.reason}
    except Exception as exc:  # noqa: BLE001
        return {"score": None, "judged_by_llm": True, "reason": f"Judge parse error: {exc}"}
