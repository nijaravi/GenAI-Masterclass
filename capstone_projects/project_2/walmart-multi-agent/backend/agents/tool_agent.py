"""
tool_agent.py — Tool-call agent with deterministic business tools.

Responsibility: Execute structured computations or data lookups using
predefined tools. In production these tools would call real APIs or
databases. Here they are implemented as pure Python functions.

Available tools:
  - calculate_price_discount(original_price, discount_pct)
  - check_inventory_level(sku_id, store_id)
  - get_billing_summary(client_id, month)
  - get_order_status(order_id)
"""
import re
from typing import Optional, Dict, Any
from backend.config import AgentStep, UserRole
from backend.utils.llm_client import run_tool_agent

# ─── Tool Implementations ────────────────────────────────────────────────────

def calculate_price_discount(original_price: float, discount_pct: float) -> Dict[str, Any]:
    discount_amount = round(original_price * discount_pct / 100, 2)
    final_price = round(original_price - discount_amount, 2)
    return {
        "original_price": original_price,
        "discount_pct": discount_pct,
        "discount_amount": discount_amount,
        "final_price": final_price,
    }


def check_inventory_level(sku_id: str, store_id: str = "WMT-001") -> Dict[str, Any]:
    # Mock inventory data
    MOCK_INVENTORY = {
        "TV-65-OLED": {"units": 47, "warehouse": 312, "status": "In Stock"},
        "AIRPODS-PRO": {"units": 8, "warehouse": 145, "status": "Low Stock"},
        "SAMSUNG-S24": {"units": 0, "warehouse": 89, "status": "Out of Stock"},
    }
    sku_upper = sku_id.upper()
    data = MOCK_INVENTORY.get(sku_upper, {"units": 22, "warehouse": 200, "status": "In Stock"})
    return {"sku_id": sku_id, "store_id": store_id, **data}


def get_billing_summary(client_id: str, month: str = "2024-11") -> Dict[str, Any]:
    return {
        "client_id": client_id,
        "month": month,
        "total_invoiced": 142_850.00,
        "total_paid": 127_300.00,
        "outstanding": 15_550.00,
        "invoices": [
            {"id": "INV-4421", "amount": 72_000.00, "status": "Paid"},
            {"id": "INV-4422", "amount": 55_300.00, "status": "Paid"},
            {"id": "INV-4423", "amount": 15_550.00, "status": "Pending"},
        ],
        "next_due_date": "2024-12-15",
    }


def get_order_status(order_id: str) -> Dict[str, Any]:
    return {
        "order_id": order_id,
        "status": "Shipped",
        "carrier": "FedEx",
        "tracking": "7489234892384923",
        "estimated_delivery": "2024-11-28",
        "items": 3,
    }


# ─── Tool Dispatcher ─────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "calculate_price_discount": calculate_price_discount,
    "check_inventory_level": check_inventory_level,
    "get_billing_summary": get_billing_summary,
    "get_order_status": get_order_status,
}


def _detect_and_run_tool(message: str) -> Optional[Dict[str, Any]]:
    """Simple keyword-based tool dispatcher for demo mode."""
    lower = message.lower()

    if re.search(r"discount|price|how much|cost", lower):
        # Try to extract numbers from message
        nums = re.findall(r"\d+\.?\d*", message)
        price = float(nums[0]) if len(nums) > 0 else 1299.0
        pct = float(nums[1]) if len(nums) > 1 else 62.0
        return calculate_price_discount(price, pct)

    if re.search(r"inventory|stock|available", lower):
        sku_match = re.search(r"\b([A-Z0-9\-]{4,20})\b", message.upper())
        sku = sku_match.group(1) if sku_match else "TV-65-OLED"
        return check_inventory_level(sku)

    if re.search(r"billing|invoice|outstanding|payment", lower):
        return get_billing_summary("CLIENT-WMT-4892")

    if re.search(r"order|tracking|shipment|delivery", lower):
        order_match = re.search(r"\b(\d{6,12})\b", message)
        order_id = order_match.group(1) if order_match else "WMT-2024-88421"
        return get_order_status(order_id)

    return None


class ToolAgent:
    """
    ToolAgent dispatches to deterministic tool functions based on the
    user's request. Tool results are formatted into a human-readable answer.
    """

    name = "ToolAgent"

    def run(self, user_message: str, user_role: UserRole) -> AgentStep:
        # Try deterministic tool dispatch first
        tool_result = _detect_and_run_tool(user_message)
        tool_name = "unknown_tool"

        if tool_result:
            # Format the tool result into a readable answer
            lower = user_message.lower()
            if "discount" in lower or "price" in lower or "cost" in lower:
                tool_name = "calculate_price_discount"
                answer = (
                    f"**Price Calculation Result:**\n"
                    f"- Original price: ${tool_result['original_price']:,.2f}\n"
                    f"- Discount: {tool_result['discount_pct']}% (${tool_result['discount_amount']:,.2f})\n"
                    f"- **Final price: ${tool_result['final_price']:,.2f}**"
                )
            elif "inventory" in lower or "stock" in lower:
                tool_name = "check_inventory_level"
                answer = (
                    f"**Inventory Check — SKU: {tool_result['sku_id']}**\n"
                    f"- Status: {tool_result['status']}\n"
                    f"- Store units: {tool_result['units']}\n"
                    f"- Warehouse units: {tool_result['warehouse']}"
                )
            elif "billing" in lower or "invoice" in lower:
                tool_name = "get_billing_summary"
                answer = (
                    f"**Billing Summary — {tool_result['month']}**\n"
                    f"- Total invoiced: ${tool_result['total_invoiced']:,.2f}\n"
                    f"- Total paid: ${tool_result['total_paid']:,.2f}\n"
                    f"- Outstanding: **${tool_result['outstanding']:,.2f}**\n"
                    f"- Next due date: {tool_result['next_due_date']}"
                )
            else:
                tool_name = "get_order_status"
                answer = (
                    f"**Order Status — #{tool_result['order_id']}**\n"
                    f"- Status: {tool_result['status']}\n"
                    f"- Carrier: {tool_result['carrier']}\n"
                    f"- Tracking: {tool_result['tracking']}\n"
                    f"- Estimated delivery: {tool_result['estimated_delivery']}"
                )
        else:
            answer = run_tool_agent(user_message)
            tool_name = "llm_tool_selection"

        return AgentStep(
            agent=self.name,
            input=user_message,
            output=answer,
            metadata={
                "tool_called": tool_name,
                "tool_result": tool_result,
            },
        )
