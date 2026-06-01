"""
src/tools/ecommerce_tools.py

Tool set: Smart E-commerce Assistant
Scenario: User can ask the agent to:
  - Check product stock
  - Get discount from coupon code
  - Calculate shipping cost
  - Get product price
  - Calculate total order cost

Each tool is a plain Python function that returns a string (the Observation).
"""

from typing import Dict, Any, List

# ─────────────────────────────────────────────────────────────────
# Mock databases (in a real system, these would be API/DB calls)
# ─────────────────────────────────────────────────────────────────
STOCK_DB: Dict[str, int] = {
    "iphone 15": 10,
    "iphone 14": 5,
    "samsung s24": 0,
    "macbook pro": 3,
    "airpods pro": 15,
    "ipad": 8,
    "laptop asus": 12,
}

PRICE_DB: Dict[str, float] = {
    "iphone 15": 25_000_000,
    "iphone 14": 18_000_000,
    "samsung s24": 22_000_000,
    "macbook pro": 45_000_000,
    "airpods pro": 7_000_000,
    "ipad": 20_000_000,
    "laptop asus": 15_000_000,
}

COUPON_DB: Dict[str, float] = {
    "WINNER": 10.0,     # 10% off
    "SALE20": 20.0,     # 20% off
    "VIP50": 50.0,      # 50% off
    "NEWUSER": 15.0,    # 15% off
}

# Shipping cost (VND) per destination
SHIPPING_DB: Dict[str, float] = {
    "hanoi":    30_000,
    "hcm":      50_000,
    "danang":   40_000,
    "cantho":   60_000,
    "hue":      45_000,
}

# ─────────────────────────────────────────────────────────────────
# Tool functions
# ─────────────────────────────────────────────────────────────────

def check_stock(item_name: str) -> str:
    """Check available stock quantity for a product."""
    key = item_name.strip().lower()
    if key in STOCK_DB:
        qty = STOCK_DB[key]
        if qty == 0:
            return f"Out of stock: '{item_name}' is currently unavailable."
        return f"In stock: '{item_name}' has {qty} units available."
    return f"Product not found: '{item_name}'. Available products: {', '.join(STOCK_DB.keys())}."


def get_product_price(item_name: str) -> str:
    """Get the unit price (in VND) of a product."""
    key = item_name.strip().lower()
    if key in PRICE_DB:
        price = PRICE_DB[key]
        return f"Unit price of '{item_name}': {price:,.0f} VND."
    return f"Price not found for '{item_name}'. Available products: {', '.join(PRICE_DB.keys())}."


def get_discount(coupon_code: str) -> str:
    """Return the discount percentage for a given coupon code."""
    code = coupon_code.strip().upper()
    if code in COUPON_DB:
        pct = COUPON_DB[code]
        return f"Coupon '{code}' is valid. Discount: {pct}% off the total product cost."
    return f"Coupon '{code}' is invalid or expired. Available coupons: {', '.join(COUPON_DB.keys())}."


def calc_shipping(weight_kg: float, destination: str) -> str:
    """
    Calculate shipping cost based on weight (kg) and destination city.
    Destinations: hanoi, hcm, danang, cantho, hue.
    """
    dest = destination.strip().lower()
    if dest not in SHIPPING_DB:
        return (
            f"Destination '{destination}' not supported. "
            f"Supported cities: {', '.join(SHIPPING_DB.keys())}."
        )
    try:
        weight = float(weight_kg)
    except (ValueError, TypeError):
        return f"Invalid weight value: '{weight_kg}'. Please provide a number."

    base_cost = SHIPPING_DB[dest]
    # Extra 10,000 VND per kg over 1 kg
    extra = max(0, weight - 1) * 10_000
    total_shipping = base_cost + extra
    return (
        f"Shipping cost to {destination} for {weight} kg: {total_shipping:,.0f} VND "
        f"(base: {base_cost:,.0f} + extra: {extra:,.0f})."
    )


def calculate_total(
    unit_price: float,
    quantity: int,
    discount_pct: float,
    shipping_cost: float,
) -> str:
    """
    Calculate the final order total.
    Formula: total = (unit_price * quantity) * (1 - discount_pct/100) + shipping_cost
    """
    try:
        unit_price = float(unit_price)
        quantity = int(quantity)
        discount_pct = float(discount_pct)
        shipping_cost = float(shipping_cost)
    except (ValueError, TypeError) as e:
        return f"Calculation error - invalid arguments: {e}"

    subtotal = unit_price * quantity
    discount_amount = subtotal * (discount_pct / 100)
    discounted_price = subtotal - discount_amount
    total = discounted_price + shipping_cost

    return (
        f"Order Summary:\n"
        f"  Subtotal ({quantity} items × {unit_price:,.0f} VND): {subtotal:,.0f} VND\n"
        f"  Discount ({discount_pct}%): -{discount_amount:,.0f} VND\n"
        f"  Shipping: {shipping_cost:,.0f} VND\n"
        f"  ─────────────────────────────\n"
        f"  TOTAL: {total:,.0f} VND"
    )


# ─────────────────────────────────────────────────────────────────
# Tool registry — maps tool name → (function, description, params hint)
# The agent uses this to dispatch calls and build the system prompt.
# ─────────────────────────────────────────────────────────────────
TOOLS_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "check_stock",
        "description": (
            "Check how many units of a product are available in stock. "
            "Input: item_name (string, e.g. 'iPhone 15'). "
            "Returns: stock count or out-of-stock message."
        ),
        "function": check_stock,
    },
    {
        "name": "get_product_price",
        "description": (
            "Get the unit price (in VND) of a product. "
            "Input: item_name (string, e.g. 'MacBook Pro'). "
            "Returns: price in VND."
        ),
        "function": get_product_price,
    },
    {
        "name": "get_discount",
        "description": (
            "Retrieve the discount percentage for a coupon code. "
            "Input: coupon_code (string, e.g. 'WINNER'). "
            "Returns: discount percentage (float) or error if invalid."
        ),
        "function": get_discount,
    },
    {
        "name": "calc_shipping",
        "description": (
            "Calculate shipping cost to a destination city. "
            "Inputs: weight_kg (float, product weight in kilograms), "
            "destination (string: 'hanoi', 'hcm', 'danang', 'cantho', 'hue'). "
            "Format: calc_shipping(weight_kg=0.5, destination=hanoi). "
            "Returns: shipping cost in VND."
        ),
        "function": calc_shipping,
    },
    {
        "name": "calculate_total",
        "description": (
            "Calculate the final order total including discount and shipping. "
            "Inputs: unit_price (float, VND), quantity (int), "
            "discount_pct (float, percentage e.g. 10 for 10%), "
            "shipping_cost (float, VND). "
            "Format: calculate_total(unit_price=25000000, quantity=2, discount_pct=10, shipping_cost=30000). "
            "Returns: itemized order summary with TOTAL in VND."
        ),
        "function": calculate_total,
    },
]
