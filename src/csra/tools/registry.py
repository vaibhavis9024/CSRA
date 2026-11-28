"""Allowlisted tool registry with argument validation and ownership scoping."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from csra.schemas.support import ToolResult
from csra.tools.investigation import (
    get_customer_orders,
    get_customer_profile,
    get_delivery_status,
    get_order_details,
    get_payment_status,
)
from csra.tools.mock_data import load_mock_store

logger = logging.getLogger(__name__)


class CustomerScopeArgs(BaseModel):
    """Arguments for customer-scoped tools (customer_id comes from request context)."""

    customer_id: str | None = None


class OrderIdArgs(BaseModel):
    """Arguments for order-scoped tools."""

    order_id: str = Field(..., min_length=1)


@dataclass(frozen=True)
class ToolSpec:
    """Metadata and handler for an allowlisted read-only tool."""

    name: str
    description: str
    arg_schema: type[BaseModel]
    requires_order_id: bool
    handler: Callable[..., ToolResult]


class ToolRegistry:
    """Controlled registry that validates and executes allowlisted investigation tools."""

    def __init__(self, store: dict[str, Any] | None = None) -> None:
        self._store = store if store is not None else load_mock_store()
        self._tools: dict[str, ToolSpec] = {
            "get_customer_profile": ToolSpec(
                name="get_customer_profile",
                description="Get basic non-sensitive profile for the current customer.",
                arg_schema=CustomerScopeArgs,
                requires_order_id=False,
                handler=lambda _args, cid: get_customer_profile(cid, self._store),
            ),
            "get_customer_orders": ToolSpec(
                name="get_customer_orders",
                description="List all orders belonging to the current customer.",
                arg_schema=CustomerScopeArgs,
                requires_order_id=False,
                handler=lambda _args, cid: get_customer_orders(cid, self._store),
            ),
            "get_order_details": ToolSpec(
                name="get_order_details",
                description="Get order items, status, date, and total for a given order_id.",
                arg_schema=OrderIdArgs,
                requires_order_id=True,
                handler=lambda args, cid: get_order_details(args.order_id, cid, self._store),
            ),
            "get_payment_status": ToolSpec(
                name="get_payment_status",
                description="Get payment status and transaction reference for a given order_id.",
                arg_schema=OrderIdArgs,
                requires_order_id=True,
                handler=lambda args, cid: get_payment_status(args.order_id, cid, self._store),
            ),
            "get_delivery_status": ToolSpec(
                name="get_delivery_status",
                description="Get shipment status, tracking, and delivery estimate for an order_id.",
                arg_schema=OrderIdArgs,
                requires_order_id=True,
                handler=lambda args, cid: get_delivery_status(args.order_id, cid, self._store),
            ),
        }

    def describe_tools(self) -> list[dict[str, str]]:
        """Return tool names and descriptions for the LLM planner prompt."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "arguments": "order_id (string)" if spec.requires_order_id else "none (uses request customer_id)",
            }
            for spec in self._tools.values()
        ]

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        request_customer_id: str,
    ) -> ToolResult:
        """Validate tool name and arguments, enforce customer scope, and execute."""
        start = time.perf_counter()
        spec = self._tools.get(tool_name)
        if spec is None:
            logger.warning("Rejected unknown tool request: %s", tool_name)
            return ToolResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error_type="unknown_tool",
                error=f"Tool '{tool_name}' is not in the allowlisted investigation tools.",
            )

        # Prevent LLM from accessing a different customer's profile/orders
        requested_cid = arguments.get("customer_id")
        if requested_cid and requested_cid != request_customer_id:
            logger.warning("Blocked cross-customer tool request for tool=%s", tool_name)
            return ToolResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error_type="access_denied",
                error="Cross-customer access is not permitted.",
            )

        try:
            validated_args = spec.arg_schema.model_validate(arguments)
        except ValidationError as exc:
            logger.warning("Invalid arguments for tool=%s: %s", tool_name, exc)
            return ToolResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error_type="invalid_arguments",
                error=f"Invalid arguments for '{tool_name}': {exc}",
            )

        try:
            result = spec.handler(validated_args, request_customer_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected failure in tool=%s", tool_name)
            result = ToolResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error_type="execution_error",
                error=f"Tool execution failed: {exc}",
            )

        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "Tool executed: name=%s success=%s error_type=%s duration_ms=%.2f",
            tool_name,
            result.success,
            result.error_type,
            duration_ms,
        )
        return result


def get_tool_registry() -> ToolRegistry:
    """Return a default ToolRegistry backed by the mock store."""
    return ToolRegistry()
