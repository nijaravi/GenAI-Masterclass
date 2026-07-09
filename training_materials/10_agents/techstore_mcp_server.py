from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Resource, Prompt, PromptMessage, TextResourceContents
import json, asyncio

server = Server("techstore")

# ── Mock data (replace with real DB calls in production) ──
PRODUCTS = {
    "laptop-pro":     {"name": "TechStore Laptop Pro",      "price": 1299.99, "qty": 23,  "category": "laptops"},
    "wireless-mouse": {"name": "ErgoClick Wireless Mouse",  "price": 49.99,  "qty": 156, "category": "accessories"},
    "usb-hub":        {"name": "USB-C Hub 7-in-1",          "price": 34.99,  "qty": 0,   "category": "accessories"},
    "monitor-4k":     {"name": "UltraView 4K Monitor 27in", "price": 449.99, "qty": 12,  "category": "monitors"},
    "keyboard-mech":  {"name": "MechType Pro Keyboard",     "price": 129.99, "qty": 67,  "category": "accessories"},
}
ORDERS = {
    "ORD-10001": {"status": "delivered",  "items": ["laptop-pro"],                 "total": 1299.99, "date": "2025-12-15"},
    "ORD-10002": {"status": "shipped",    "items": ["wireless-mouse", "usb-hub"],  "total": 84.98,  "date": "2026-01-03"},
    "ORD-10003": {"status": "processing", "items": ["monitor-4k"],                 "total": 449.99, "date": "2026-01-10"},
}

# ── Tool 1: Check inventory ──
@server.tool()
async def check_inventory(product_name: str) -> list[TextContent]:
    """Check if a product is in stock and return its price.

    Use when a customer asks about product availability, stock levels,
    or pricing for a specific item. Accepts partial names (e.g. 'laptop').
    """
    matches = [p for p in PRODUCTS.values() if product_name.lower() in p["name"].lower()]
    if not matches:
        return [TextContent(type="text",
            text=f"No products matching '{product_name}'. Available: laptops, accessories, monitors.")]
    lines = [
        f"  {p['name']}: ${p['price']:.2f} — {'In Stock' if p['qty'] > 0 else 'OUT OF STOCK'} ({p['qty']} units)"
        for p in matches
    ]
    return [TextContent(type="text", text="\n".join(lines))]

# ── Tool 2: Search products ──
@server.tool()
async def search_products(query: str = "", category: str = "", in_stock_only: bool = False) -> list[TextContent]:
    """Search the product catalog by name, category, or availability.

    Use when a customer wants to browse products or find options in a category.
    Category options: laptops, accessories, monitors.
    """
    results = list(PRODUCTS.values())
    if query:
        results = [p for p in results if query.lower() in p["name"].lower()]
    if category:
        results = [p for p in results if p["category"] == category.lower()]
    if in_stock_only:
        results = [p for p in results if p["qty"] > 0]
    if not results:
        return [TextContent(type="text", text="No products found matching your search.")]
    lines = [f"Found {len(results)} product(s):"] + [
        f"  {p['name']} — ${p['price']:.2f} ({'In Stock' if p['qty'] > 0 else 'Out of Stock'})"
        for p in results
    ]
    return [TextContent(type="text", text="\n".join(lines))]

# ── Tool 3: Order status ──
@server.tool()
async def get_order_status(order_id: str) -> list[TextContent]:
    """Look up the current status of a customer order.

    Use when a customer asks about their order, delivery, or shipment.
    Order IDs have the format ORD-XXXXX.
    """
    order_id = order_id.upper().strip()
    order = ORDERS.get(order_id)
    if not order:
        return [TextContent(type="text",
            text=f"Order {order_id} not found. Please verify the ID (format: ORD-XXXXX).")]
    items_str = ", ".join(PRODUCTS.get(i, {"name": i})["name"] for i in order["items"])
    return [TextContent(type="text",
        text=f"Order {order_id}:\n"
             f"  Status: {order['status'].upper()}\n"
             f"  Items: {items_str}\n"
             f"  Total: ${order['total']:.2f}\n"
             f"  Date:  {order['date']}")]

# ── Resource: Refund policy document ──
@server.list_resources()
async def list_resources():
    return [Resource(uri="docs://refund-policy", name="Refund Policy", mimeType="text/markdown")]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "docs://refund-policy":
        content = (
            "# TechStore Refund Policy\n\n"
            "Returns accepted within 30 days of delivery.\n"
            "Items must be in original packaging with all accessories.\n"
            "Refund processed within 5-7 business days to original payment method.\n"
            "Digital products: no refunds after download or activation.\n"
            "Shipping costs are non-refundable unless the item was defective.\n"
        )
        return [TextResourceContents(uri=uri, mimeType="text/markdown", text=content)]
    raise ValueError(f"Unknown resource: {uri}")

# ── Prompt: Reusable order analysis template ──
@server.list_prompts()
async def list_prompts():
    return [Prompt(
        name="analyze_order",
        description="Analyse an order for potential issues or anomalies",
        arguments=[{"name": "order_id", "description": "Order ID to analyse", "required": True}],
    )]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict):
    if name == "analyze_order":
        order_id = arguments.get("order_id", "")
        order = ORDERS.get(order_id.upper(), {"status": "unknown", "items": [], "total": 0})
        return [PromptMessage(
            role="user",
            content=TextContent(type="text",
                text=f"Analyse order {order_id}:\n{json.dumps(order, indent=2)}\n\n"
                     f"Check for: unusual quantities, price anomalies, delivery risk.\n"
                     f"Rate each risk LOW / MEDIUM / HIGH."),
        )]
    raise ValueError(f"Unknown prompt: {name}")

# ── Entry point ──
if __name__ == "__main__":
    async def main():
        async with stdio_server() as (r, w):
            await server.run(r, w)
    asyncio.run(main())
