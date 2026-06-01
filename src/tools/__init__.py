"""
src/tools/__init__.py

Tool Registry — exports all available tools for the ReAct Agent.

Tool files in this package:
  - ecommerce_tools.py     : ORDER tools (stock, price, discount, shipping, total)
  - recommendation_tools.py: DISCOVERY tools (budget search, compare, promotions, alternatives)

Usage:
    from src.tools import TOOLS_REGISTRY          # all tools combined
    from src.tools import ECOMMERCE_TOOLS         # order tools only
    from src.tools import RECOMMENDATION_TOOLS    # discovery tools only
"""

from src.tools.ecommerce_tools import (
    check_stock,
    get_product_price,
    get_discount,
    calc_shipping,
    calculate_total,
    TOOLS_REGISTRY as ECOMMERCE_TOOLS,
)

from src.tools.recommendation_tools import (
    find_products_by_budget,
    compare_products,
    get_active_promotions,
    find_alternative_product,
    RECOMMENDATION_TOOLS_REGISTRY as RECOMMENDATION_TOOLS,
)

# Combined registry — all 9 tools available to the Agent
TOOLS_REGISTRY = ECOMMERCE_TOOLS + RECOMMENDATION_TOOLS

__all__ = [
    # Order tools
    "check_stock",
    "get_product_price",
    "get_discount",
    "calc_shipping",
    "calculate_total",
    "ECOMMERCE_TOOLS",
    # Recommendation tools
    "find_products_by_budget",
    "compare_products",
    "get_active_promotions",
    "find_alternative_product",
    "RECOMMENDATION_TOOLS",
    # Combined
    "TOOLS_REGISTRY",
]
