"""Unit tests for Stage 1 read-only investigation tools and ToolRegistry."""

from __future__ import annotations

from csra.tools.registry import ToolRegistry


def test_valid_customer_and_order_lookups() -> None:
    reg = ToolRegistry()

    prof = reg.execute("get_customer_profile", {}, "customer_101")
    assert prof.success is True
    assert prof.data is not None
    assert prof.data["name"] == "Maya Lin"

    orders = reg.execute("get_customer_orders", {}, "customer_101")
    assert orders.success is True
    assert orders.data is not None
    assert orders.data["order_count"] == 2

    no_orders = reg.execute("get_customer_orders", {}, "customer_103")
    assert no_orders.success is True
    assert no_orders.data is not None
    assert no_orders.data["order_count"] == 0

    details = reg.execute("get_order_details", {"order_id": "ORD-1001"}, "customer_101")
    assert details.success is True
    assert details.data is not None
    assert details.data["order_status"] == "unpaid"

    pay = reg.execute("get_payment_status", {"order_id": "ORD-1001"}, "customer_101")
    assert pay.success is True
    assert pay.data is not None
    assert pay.data["payment_status"] == "deducted"

    deliv = reg.execute("get_delivery_status", {"order_id": "ORD-1003"}, "customer_102")
    assert deliv.success is True
    assert deliv.data is not None
    assert deliv.data["delivery_exception"] == "weather_delay"


def test_unknown_customer_and_unknown_order() -> None:
    reg = ToolRegistry()

    unk_cust = reg.execute("get_customer_profile", {}, "customer_000")
    assert unk_cust.success is False
    assert unk_cust.error_type == "not_found"

    unk_ord = reg.execute("get_order_details", {"order_id": "ORD-0000"}, "customer_101")
    assert unk_ord.success is False
    assert unk_ord.error_type == "not_found"


def test_ownership_enforcement_blocks_other_customer_order() -> None:
    reg = ToolRegistry()

    denied = reg.execute("get_order_details", {"order_id": "ORD-9999"}, "customer_101")
    assert denied.success is False
    assert denied.error_type == "access_denied"
    assert "customer_999" not in (denied.error or "")

    cross_cust = reg.execute("get_customer_orders", {"customer_id": "customer_102"}, "customer_101")
    assert cross_cust.success is False
    assert cross_cust.error_type == "access_denied"


def test_invalid_args_unknown_tool_and_execution_error() -> None:
    reg = ToolRegistry()

    unknown = reg.execute("issue_refund", {"order_id": "ORD-1001"}, "customer_101")
    assert unknown.success is False
    assert unknown.error_type == "unknown_tool"

    bad_args = reg.execute("get_order_details", {"order_id": ""}, "customer_101")
    assert bad_args.success is False
    assert bad_args.error_type == "invalid_arguments"

    # Simulate unexpected tool failure
    broken_reg = ToolRegistry(store={"orders": None})  # type: ignore[dict-item]
    failed = broken_reg.execute("get_order_details", {"order_id": "ORD-1001"}, "customer_101")
    assert failed.success is False
    assert failed.error_type == "execution_error"
