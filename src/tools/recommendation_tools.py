"""
src/tools/recommendation_tools.py

Tool set #2: Product Recommendation & Comparison Tools

PURPOSE:
    These tools extend the Agent's capability beyond basic stock/price queries.
    They allow the Agent to help users:
      - Find products within their budget
      - Compare two products side-by-side
      - Get current active promotions
      - Find alternative products when something is out of stock

WHY SEPARATE FROM ecommerce_tools.py:
    - Different concern: ecommerce_tools.py handles ORDER operations
      (stock, price, discount, shipping, total calculation)
    - This file handles DISCOVERY operations
      (search, filter, compare, recommend)
    - Separation of concerns makes each tool file focused and testable

DESIGN PRINCIPLE:
    Each tool returns a human-readable string so the LLM can directly
    embed the result as an "Observation" in the ReAct loop without
    any additional parsing.
"""

from typing import Dict, Any, List, Optional

# ─────────────────────────────────────────────────────────────────
# Shared product catalog (mirrors ecommerce_tools.py databases)
# In production, both files would import from a shared data layer.
# ─────────────────────────────────────────────────────────────────
PRODUCT_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "iPhone 15",
        "category": "smartphone",
        "brand": "Apple",
        "price_vnd": 25_000_000,
        "stock": 10,
        "rating": 4.8,
        "specs": {
            "display": "6.1 inch Super Retina XDR",
            "chip": "A16 Bionic",
            "camera": "48MP + 12MP dual camera",
            "battery": "3,349 mAh",
            "storage_options": ["128GB", "256GB", "512GB"],
        },
        "tags": ["apple", "phone", "premium", "5g"],
    },
    {
        "name": "iPhone 14",
        "category": "smartphone",
        "brand": "Apple",
        "price_vnd": 18_000_000,
        "stock": 5,
        "rating": 4.6,
        "specs": {
            "display": "6.1 inch Super Retina XDR",
            "chip": "A15 Bionic",
            "camera": "12MP + 12MP dual camera",
            "battery": "3,279 mAh",
            "storage_options": ["128GB", "256GB", "512GB"],
        },
        "tags": ["apple", "phone", "mid-range", "5g"],
    },
    {
        "name": "Samsung S24",
        "category": "smartphone",
        "brand": "Samsung",
        "price_vnd": 22_000_000,
        "stock": 0,
        "rating": 4.7,
        "specs": {
            "display": "6.2 inch Dynamic AMOLED 2X",
            "chip": "Snapdragon 8 Gen 3",
            "camera": "50MP + 12MP + 10MP triple camera",
            "battery": "4,000 mAh",
            "storage_options": ["256GB"],
        },
        "tags": ["samsung", "phone", "android", "premium", "5g"],
    },
    {
        "name": "MacBook Pro",
        "category": "laptop",
        "brand": "Apple",
        "price_vnd": 45_000_000,
        "stock": 3,
        "rating": 4.9,
        "specs": {
            "display": "14.2 inch Liquid Retina XDR",
            "chip": "Apple M3 Pro",
            "ram": "18GB unified memory",
            "storage": "512GB SSD",
            "battery": "Up to 18 hours",
        },
        "tags": ["apple", "laptop", "pro", "creative", "premium"],
    },
    {
        "name": "AirPods Pro",
        "category": "audio",
        "brand": "Apple",
        "price_vnd": 7_000_000,
        "stock": 15,
        "rating": 4.7,
        "specs": {
            "type": "True Wireless Earbuds",
            "anc": "Active Noise Cancellation",
            "battery": "6 hrs (30 hrs with case)",
            "chip": "H2",
            "connectivity": "Bluetooth 5.3",
        },
        "tags": ["apple", "audio", "earbuds", "anc", "wireless"],
    },
    {
        "name": "iPad",
        "category": "tablet",
        "brand": "Apple",
        "price_vnd": 20_000_000,
        "stock": 8,
        "rating": 4.6,
        "specs": {
            "display": "10.9 inch Liquid Retina",
            "chip": "A14 Bionic",
            "camera": "12MP ultrawide front",
            "storage_options": ["64GB", "256GB"],
            "connectivity": "Wi-Fi 6 + optional 5G",
        },
        "tags": ["apple", "tablet", "education", "creative"],
    },
    {
        "name": "Laptop Asus",
        "category": "laptop",
        "brand": "Asus",
        "price_vnd": 15_000_000,
        "stock": 12,
        "rating": 4.3,
        "specs": {
            "display": "15.6 inch Full HD IPS",
            "cpu": "Intel Core i5-12th Gen",
            "ram": "16GB DDR4",
            "storage": "512GB SSD",
            "battery": "Up to 8 hours",
        },
        "tags": ["asus", "laptop", "budget", "student", "office"],
    },
]

# Active promotions catalog
PROMOTIONS: List[Dict[str, Any]] = [
    {
        "name": "Summer Sale 2026",
        "code": "SALE20",
        "discount_pct": 20,
        "applies_to": ["iphone 14", "laptop asus", "ipad"],
        "valid_until": "2026-06-30",
        "description": "Summer sale — 20% off on selected products.",
    },
    {
        "name": "Welcome Offer",
        "code": "NEWUSER",
        "discount_pct": 15,
        "applies_to": "all",
        "valid_until": "2026-12-31",
        "description": "15% off for new customers on any product.",
    },
    {
        "name": "Flash Deal",
        "code": "WINNER",
        "discount_pct": 10,
        "applies_to": "all",
        "valid_until": "2026-06-30",
        "description": "Flash deal — 10% off any product, limited time.",
    },
    {
        "name": "VIP Exclusive",
        "code": "VIP50",
        "discount_pct": 50,
        "applies_to": ["macbook pro", "iphone 15", "samsung s24"],
        "valid_until": "2026-06-15",
        "description": "Exclusive 50% off for VIP members on premium items.",
    },
]


# ─────────────────────────────────────────────────────────────────
# Tool 1: find_products_by_budget
# ─────────────────────────────────────────────────────────────────
def find_products_by_budget(max_price_vnd: float, category: Optional[str] = None) -> str:
    """
    Find all in-stock products within a given budget (in VND).

    Args:
        max_price_vnd: Maximum price the user is willing to pay (VND).
        category     : Optional filter — 'smartphone', 'laptop', 'audio', 'tablet'.
                       If not specified, search all categories.

    Returns:
        Formatted list of matching products sorted by price descending
        (best value first), or a message if nothing matches.
    """
    try:
        budget = float(max_price_vnd)
    except (ValueError, TypeError):
        return f"Invalid budget value: '{max_price_vnd}'. Please provide a number."

    results = []
    for product in PRODUCT_CATALOG:
        # Skip out-of-stock products
        if product["stock"] == 0:
            continue
        # Apply budget filter
        if product["price_vnd"] > budget:
            continue
        # Apply category filter if provided
        if category:
            cat = category.strip().lower()
            if product["category"] != cat:
                continue
        results.append(product)

    if not results:
        cat_msg = f" in category '{category}'" if category else ""
        return (
            f"No in-stock products found{cat_msg} within budget "
            f"{budget:,.0f} VND. "
            f"Try increasing your budget or removing the category filter."
        )

    # Sort by price descending (best/most featured first)
    results.sort(key=lambda p: p["price_vnd"], reverse=True)

    lines = [f"Products within budget {budget:,.0f} VND:"]
    for p in results:
        lines.append(
            f"  • {p['name']} ({p['brand']}) — "
            f"{p['price_vnd']:,.0f} VND | "
            f"Rating: {p['rating']}/5 | "
            f"Stock: {p['stock']} units"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Tool 2: compare_products
# ─────────────────────────────────────────────────────────────────
def compare_products(product_a: str, product_b: str) -> str:
    """
    Compare two products side-by-side: price, rating, stock, and key specs.

    Args:
        product_a: Name of first product (e.g. 'iPhone 15').
        product_b: Name of second product (e.g. 'Samsung S24').

    Returns:
        A formatted comparison table with verdict on which is better value.
    """
    def _find(name: str) -> Optional[Dict]:
        key = name.strip().lower()
        for p in PRODUCT_CATALOG:
            if p["name"].lower() == key:
                return p
        return None

    pa = _find(product_a)
    pb = _find(product_b)

    missing = []
    if pa is None:
        missing.append(product_a)
    if pb is None:
        missing.append(product_b)
    if missing:
        available = ", ".join(p["name"] for p in PRODUCT_CATALOG)
        return (
            f"Product(s) not found: {', '.join(missing)}. "
            f"Available: {available}."
        )

    # Build comparison
    lines = [
        f"Comparison: {pa['name']} vs {pb['name']}",
        f"{'Attribute':<20} {'  ' + pa['name']:<25} {'  ' + pb['name']:<25}",
        "─" * 70,
        f"{'Price (VND)':<20} {pa['price_vnd']:>20,.0f}   {pb['price_vnd']:>20,.0f}",
        f"{'Rating':<20} {pa['rating']:>20}/5       {pb['rating']:>20}/5",
        f"{'Stock':<20} {pa['stock']:>18} units   {pb['stock']:>18} units",
        f"{'Category':<20} {pa['category']:>20}   {pb['category']:>20}",
        f"{'Brand':<20} {pa['brand']:>20}   {pb['brand']:>20}",
        "─" * 70,
    ]

    # Add specs comparison (common keys only)
    specs_a = pa.get("specs", {})
    specs_b = pb.get("specs", {})
    all_spec_keys = set(specs_a.keys()) | set(specs_b.keys())
    for key in sorted(all_spec_keys):
        val_a = str(specs_a.get(key, "N/A"))
        val_b = str(specs_b.get(key, "N/A"))
        lines.append(f"{key:<20} {val_a:<25} {val_b:<25}")

    lines.append("─" * 70)

    # Verdict
    if pa["price_vnd"] < pb["price_vnd"] and pa["rating"] >= pb["rating"]:
        verdict = f"Verdict: {pa['name']} offers better value (cheaper + same/better rating)."
    elif pb["price_vnd"] < pa["price_vnd"] and pb["rating"] >= pa["rating"]:
        verdict = f"Verdict: {pb['name']} offers better value (cheaper + same/better rating)."
    elif pa["rating"] > pb["rating"]:
        verdict = f"Verdict: {pa['name']} has higher rating but costs more."
    else:
        verdict = f"Verdict: Both products are similar in value. Choose based on brand preference."

    lines.append(verdict)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Tool 3: get_active_promotions
# ─────────────────────────────────────────────────────────────────
def get_active_promotions(product_name: Optional[str] = None) -> str:
    """
    List all current active promotions, optionally filtered by product.

    Args:
        product_name: Optional — filter promotions applicable to a specific product.
                      If not provided, returns all active promotions.

    Returns:
        Formatted list of promotions with code, discount, and eligibility.
    """
    if product_name:
        key = product_name.strip().lower()
        relevant = []
        for promo in PROMOTIONS:
            applies = promo.get("applies_to", [])
            if applies == "all" or key in applies:
                relevant.append(promo)

        if not relevant:
            return (
                f"No specific promotions found for '{product_name}'. "
                f"However, check general promotions — some apply to all products."
            )

        lines = [f"Active promotions applicable to '{product_name}':"]
        for p in relevant:
            lines.append(
                f"  • [{p['code']}] {p['name']} — "
                f"{p['discount_pct']}% off | "
                f"Valid until: {p['valid_until']} | "
                f"{p['description']}"
            )
        return "\n".join(lines)

    # All promotions
    lines = ["All active promotions:"]
    for p in PROMOTIONS:
        applies_str = (
            "All products"
            if p["applies_to"] == "all"
            else ", ".join(str(x).title() for x in p["applies_to"])
        )
        lines.append(
            f"  • [{p['code']}] {p['name']} — "
            f"{p['discount_pct']}% off | "
            f"Applies to: {applies_str} | "
            f"Valid until: {p['valid_until']}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Tool 4: find_alternative_product
# ─────────────────────────────────────────────────────────────────
def find_alternative_product(product_name: str) -> str:
    """
    Suggest in-stock alternatives when a requested product is unavailable.

    Args:
        product_name: Name of the out-of-stock or unavailable product.

    Returns:
        Up to 3 in-stock alternatives in the same category, sorted by rating.
    """
    key = product_name.strip().lower()

    # Find the original product to get its category
    original = None
    for p in PRODUCT_CATALOG:
        if p["name"].lower() == key:
            original = p
            break

    if original is None:
        return (
            f"Product '{product_name}' not found in catalog. "
            f"Available products: {', '.join(p['name'] for p in PRODUCT_CATALOG)}."
        )

    # Find in-stock alternatives in the same category (excluding itself)
    alternatives = [
        p for p in PRODUCT_CATALOG
        if p["category"] == original["category"]
        and p["name"].lower() != key
        and p["stock"] > 0
    ]

    if not alternatives:
        return (
            f"No in-stock alternatives found for '{product_name}' "
            f"in category '{original['category']}'. "
            f"Please check back later."
        )

    # Sort by rating descending, return top 3
    alternatives.sort(key=lambda p: p["rating"], reverse=True)
    top = alternatives[:3]

    lines = [
        f"Alternatives for '{product_name}' (category: {original['category']}):"
    ]
    for p in top:
        price_diff = p["price_vnd"] - original["price_vnd"]
        diff_str = (
            f"+{price_diff:,.0f} VND more"
            if price_diff > 0
            else f"{abs(price_diff):,.0f} VND cheaper"
        )
        lines.append(
            f"  • {p['name']} ({p['brand']}) — "
            f"{p['price_vnd']:,.0f} VND ({diff_str}) | "
            f"Rating: {p['rating']}/5 | Stock: {p['stock']} units"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Tool registry for this file
# ─────────────────────────────────────────────────────────────────
RECOMMENDATION_TOOLS_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "find_products_by_budget",
        "description": (
            "Find all in-stock products within a given budget. "
            "Inputs: max_price_vnd (float, maximum budget in VND), "
            "category (optional string: 'smartphone', 'laptop', 'audio', 'tablet'). "
            "Format: find_products_by_budget(max_price_vnd=20000000, category=smartphone). "
            "Returns: list of matching products with price, rating, and stock."
        ),
        "function": find_products_by_budget,
    },
    {
        "name": "compare_products",
        "description": (
            "Compare two products side-by-side: price, rating, stock, and specs. "
            "Inputs: product_a (string, e.g. 'iPhone 15'), product_b (string, e.g. 'Samsung S24'). "
            "Format: compare_products(product_a=iPhone 15, product_b=Samsung S24). "
            "Returns: side-by-side comparison table with a value verdict."
        ),
        "function": compare_products,
    },
    {
        "name": "get_active_promotions",
        "description": (
            "List all current active promotions and coupon codes. "
            "Input: product_name (optional string — filter promotions for a specific product). "
            "Format: get_active_promotions() or get_active_promotions(product_name=iPhone 15). "
            "Returns: list of promotions with code, discount percentage, and validity."
        ),
        "function": get_active_promotions,
    },
    {
        "name": "find_alternative_product",
        "description": (
            "Find in-stock alternatives when a product is out-of-stock or unavailable. "
            "Input: product_name (string, e.g. 'Samsung S24'). "
            "Format: find_alternative_product(product_name=Samsung S24). "
            "Returns: up to 3 similar in-stock products in the same category, sorted by rating."
        ),
        "function": find_alternative_product,
    },
]
