"""Deterministic read-only investigation tools with customer ownership enforcement."""

from __future__ import annotations

from typing import Any

from csra.schemas.support import ToolResult


def _verify_order_ownership(
    order_id: str,
    customer_id: str,
    store: dict[str, Any],
) -> tuple[dict[str, Any] | None, ToolResult | None]:
    """Return (order, None) if owned by customer_id, else (None, error ToolResult)."""
    order = store.get("orders", {}).get(order_id)
    if order is None:
        return None, ToolResult(
            tool_name="order_lookup",
            arguments={"order_id": order_id},
            success=False,
            error_type="not_found",
            error=f"Order '{order_id}' was not found.",
        )
    if order.get("customer_id") != customer_id:
        return None, ToolResult(
            tool_name="order_lookup",
            arguments={"order_id": order_id},
            success=False,
            error_type="access_denied",
            error=f"Order '{order_id}' is not accessible for this customer.",
        )
    return order, None


def get_customer_profile(customer_id: str, store: dict[str, Any]) -> ToolResult:
    """Return non-sensitive customer profile details."""
    customer = store.get("customers", {}).get(customer_id)
    if customer is None:
        return ToolResult(
            tool_name="get_customer_profile",
            arguments={"customer_id": customer_id},
            success=False,
            error_type="not_found",
            error=f"Customer '{customer_id}' was not found.",
        )
    return ToolResult(
        tool_name="get_customer_profile",
        arguments={"customer_id": customer_id},
        success=True,
        data=dict(customer),
    )


def get_customer_orders(customer_id: str, store: dict[str, Any]) -> ToolResult:
    """Return all orders belonging to customer_id."""
    if customer_id not in store.get("customers", {}):
        return ToolResult(
            tool_name="get_customer_orders",
            arguments={"customer_id": customer_id},
            success=False,
            error_type="not_found",
            error=f"Customer '{customer_id}' was not found.",
        )
    orders = [
        {
            "order_id": o["order_id"],
            "order_status": o["order_status"],
            "order_date": o["order_date"],
            "total_amount": o["total_amount"],
        }
        for o in store.get("orders", {}).values()
        if o.get("customer_id") == customer_id
    ]
    return ToolResult(
        tool_name="get_customer_orders",
        arguments={"customer_id": customer_id},
        success=True,
        data={"customer_id": customer_id, "orders": orders, "order_count": len(orders)},
    )


def get_order_details(order_id: str, customer_id: str, store: dict[str, Any]) -> ToolResult:
    """Return full order details if owned by customer_id."""
    order, err = _verify_order_ownership(order_id, customer_id, store)
    if err is not None:
        return err.model_copy(update={"tool_name": "get_order_details"})
    return ToolResult(
        tool_name="get_order_details",
        arguments={"order_id": order_id},
        success=True,
        data=dict(order or {}),
    )


def get_payment_status(order_id: str, customer_id: str, store: dict[str, Any]) -> ToolResult:
    """Return payment status for an order owned by customer_id."""
    _, err = _verify_order_ownership(order_id, customer_id, store)
    if err is not None:
        return err.model_copy(update={"tool_name": "get_payment_status"})

    payment = store.get("payments", {}).get(order_id)
    if payment is None:
        return ToolResult(
            tool_name="get_payment_status",
            arguments={"order_id": order_id},
            success=False,
            error_type="not_found",
            error=f"No payment record found for order '{order_id}'.",
        )
    return ToolResult(
        tool_name="get_payment_status",
        arguments={"order_id": order_id},
        success=True,
        data=dict(payment),
    )


def get_delivery_status(order_id: str, customer_id: str, store: dict[str, Any]) -> ToolResult:
    """Return shipment and delivery status for an order owned by customer_id."""
    _, err = _verify_order_ownership(order_id, customer_id, store)
    if err is not None:
        return err.model_copy(update={"tool_name": "get_delivery_status"})

    delivery = store.get("deliveries", {}).get(order_id)
    if delivery is None:
        return ToolResult(
            tool_name="get_delivery_status",
            arguments={"order_id": order_id},
            success=False,
            error_type="not_found",
            error=f"No delivery record found for order '{order_id}'.",
        )
    return ToolResult(
        tool_name="get_delivery_status",
        arguments={"order_id": order_id},
        success=True,
        data=dict(delivery),
    )
