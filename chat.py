"""
chat.py — Interactive Chat with ReAct Agent

Gõ câu hỏi bất kỳ, Agent sẽ suy nghĩ và trả lời.

Chạy:  python chat.py
       python chat.py --chatbot      (so sánh với chatbot không có tool)
       python chat.py --v2           (dùng Agent v2 với guardrails)

Commands đặc biệt:
  /help    — xem danh sách tools
  /tools   — xem 9 tools và mô tả ngắn
  /db      — xem mock database (sản phẩm, giá, coupon, ship)
  /clear   — xóa lịch sử hội thoại của agent
  /quit    — thoát
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

# ── Colors ──────────────────────────────────────────────────────
CYAN   = "\033[96m"; YELLOW = "\033[93m"; GREEN = "\033[92m"
MAGENTA= "\033[95m"; RED    = "\033[91m"; BOLD  = "\033[1m"
RESET  = "\033[0m";  DIM    = "\033[2m"
def c(t, col): return f"{col}{t}{RESET}"

# ── Mock DB quick-reference ──────────────────────────────────────
DB_INFO = """
┌─────────────────────────────────────────────────────────┐
│  📦 SẢN PHẨM           GIÁ (VND)     TỒN KHO           │
├─────────────────────────────────────────────────────────┤
│  iPhone 15             25,000,000    10 units            │
│  iPhone 14             18,000,000     5 units            │
│  Samsung S24           22,000,000     0 (HẾT HÀNG)      │
│  MacBook Pro           45,000,000     3 units            │
│  AirPods Pro            7,000,000    15 units            │
│  iPad                  20,000,000     8 units            │
│  Laptop Asus           15,000,000    12 units            │
├─────────────────────────────────────────────────────────┤
│  🎟️  COUPON             GIẢM GIÁ                         │
├─────────────────────────────────────────────────────────┤
│  WINNER                10% off       Áp dụng tất cả     │
│  SALE20                20% off       iPhone 14, Asus...  │
│  VIP50                 50% off       iPhone 15, MacBook  │
│  NEWUSER               15% off       Áp dụng tất cả     │
├─────────────────────────────────────────────────────────┤
│  🚚 SHIPPING            PHÍ (VND)                        │
├─────────────────────────────────────────────────────────┤
│  Hà Nội (hanoi)         30,000                          │
│  TP.HCM (hcm)           50,000                          │
│  Đà Nẵng (danang)       40,000                          │
│  Cần Thơ (cantho)       60,000                          │
│  Huế (hue)              45,000                          │
└─────────────────────────────────────────────────────────┘"""

TOOLS_INFO = """
  ORDER TOOLS (src/tools/ecommerce_tools.py):
    check_stock(item_name)                       — kiểm tra tồn kho
    get_product_price(item_name)                 — lấy đơn giá
    get_discount(coupon_code)                    — xác thực coupon
    calc_shipping(weight_kg, destination)        — tính phí ship
    calculate_total(unit_price, quantity,        — tính tổng đơn
                    discount_pct, shipping_cost)

  DISCOVERY TOOLS (src/tools/recommendation_tools.py):
    find_products_by_budget(max_price_vnd,       — tìm theo ngân sách
                            category?)
    compare_products(product_a, product_b)       — so sánh 2 sản phẩm
    get_active_promotions(product_name?)         — xem khuyến mãi
    find_alternative_product(product_name)       — gợi ý thay thế"""

HELP_MSG = """
  /help    — hiện menu này
  /tools   — danh sách 9 tools
  /db      — xem database sản phẩm, coupon, shipping
  /clear   — reset lịch sử agent (bắt đầu hội thoại mới)
  /quit    — thoát chương trình

  Câu hỏi mẫu:
    "Is the iPhone 15 in stock?"
    "How much does the MacBook Pro cost?"
    "Buy 2 iPhone 15 with coupon WINNER, ship to Hanoi. Total?"
    "I have coupon FAKE99 for AirPods Pro, ship to HCM?"
    "What smartphones can I buy under 20 million VND?"
    "Compare iPhone 15 and Samsung S24"
    "What promotions are available for iPhone 15?"
    "Samsung S24 is out of stock, what else can I buy?" """


def build_agent(mode: str):
    """Build agent or chatbot based on mode."""
    from src.core.openai_provider import OpenAIProvider
    from src.tools import TOOLS_REGISTRY

    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("DEFAULT_MODEL", "gpt-4o")
    llm = OpenAIProvider(model_name=model, api_key=api_key)

    if mode == "chatbot":
        from src.chatbot import SimpleChatbot
        return SimpleChatbot(llm=llm), "chatbot", llm
    elif mode == "v2":
        from src.agent.guardrails import RobustReActAgent
        return RobustReActAgent(llm=llm, tools=TOOLS_REGISTRY,
                                max_steps=6, max_retries=3), "agent_v2", llm
    else:
        from src.agent.agent import ReActAgent
        return ReActAgent(llm=llm, tools=TOOLS_REGISTRY, max_steps=6), "agent_v1", llm


def run_with_trace(agent, question: str, mode: str) -> str:
    """Run agent and print live step trace."""
    if mode == "chatbot":
        t0 = time.time()
        ans = agent.chat(question)
        ms  = int((time.time()-t0)*1000)
        print(c(f"  ⏱  {ms}ms | 1 LLM call (no tools)", DIM))
        return ans

    # For ReActAgent / RobustReActAgent — trace each step
    from src.agent.agent import _parse_action, _parse_final_answer

    conversation = f"User Question: {question}\n\n"
    system_prompt = agent.get_system_prompt()
    step = 0
    t_total = time.time()

    while step < agent.max_steps:
        step += 1
        t0 = time.time()
        result = agent.llm.generate(prompt=conversation, system_prompt=system_prompt)
        ms = int((time.time()-t0)*1000)
        llm_output = result.get("content","")
        tokens = result.get("usage",{}).get("total_tokens","?")
        conversation += llm_output + "\n"

        # Print thought/action/final
        for line in llm_output.strip().split("\n"):
            line = line.strip()
            if line.lower().startswith("thought:"):
                print(c(f"  💭 {line}", CYAN))
            elif line.lower().startswith("action:"):
                print(c(f"  ⚡ {line}", MAGENTA))
            elif line.lower().startswith("final answer:"):
                print(c(f"  ✅ {line}", GREEN))
        print(c(f"  {DIM}[Step {step} | {ms}ms | {tokens} tokens]{RESET}", DIM))

        final = _parse_final_answer(llm_output)
        if final:
            total_ms = int((time.time()-t_total)*1000)
            print(c(f"\n  ⏱  Tổng: {total_ms}ms | {step} LLM calls | {step-1} tool calls", DIM))
            return final

        action = _parse_action(llm_output)
        if action:
            tool_name, raw_args = action
            obs = agent._execute_tool(tool_name, raw_args)
            print(c(f"  🔧 → {tool_name}({raw_args[:50]})", MAGENTA))
            print(c(f"  📋   {obs[:80]}{'...' if len(obs)>80 else ''}", CYAN))
            conversation += f"Observation: {obs}\n\n"
        else:
            conversation += "Observation: No action. Please call a tool or write Final Answer.\n\n"

    return f"Could not complete within {agent.max_steps} steps."


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot", action="store_true", help="Dùng Chatbot (không có tool)")
    parser.add_argument("--v2",      action="store_true", help="Dùng Agent v2 (guardrails)")
    args = parser.parse_args()

    mode = "chatbot" if args.chatbot else ("v2" if args.v2 else "agent_v1")
    agent, mode_label, llm = build_agent(mode)

    # Header
    model = os.getenv("DEFAULT_MODEL", "gpt-4o")
    label_map = {
        "chatbot":  c("📢 Chatbot (NO tools — sẽ hallucinate!)", YELLOW),
        "agent_v1": c("🤖 Agent v1 (ReAct + 9 tools)", GREEN),
        "agent_v2": c("🛡️  Agent v2 (ReAct + Guardrails + Retry)", GREEN),
    }
    print(c(f"\n{'═'*60}", BOLD))
    print(c(f"  E-Commerce Assistant — {model}", BOLD))
    print(f"  Mode: {label_map[mode]}")
    print(c(f"{'═'*60}", BOLD))
    print(c("  Gõ /help để xem hướng dẫn | /db để xem sản phẩm\n", DIM))

    while True:
        try:
            user_input = input(c("  Bạn: ", BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print(c("\n\n  👋 Thoát. Tạm biệt!\n", GREEN))
            break

        if not user_input:
            continue

        # Special commands
        if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
            print(c("\n  👋 Tạm biệt!\n", GREEN))
            break
        elif user_input.lower() == "/help":
            print(c(HELP_MSG, CYAN))
            continue
        elif user_input.lower() == "/tools":
            print(c(TOOLS_INFO, CYAN))
            continue
        elif user_input.lower() == "/db":
            print(c(DB_INFO, CYAN))
            continue
        elif user_input.lower() == "/clear":
            if hasattr(agent, 'history'):
                agent.history.clear()
            print(c("  ✅ Đã reset lịch sử hội thoại.\n", GREEN))
            continue

        # Run agent/chatbot
        print(c(f"\n  {'─'*55}", DIM))
        answer = run_with_trace(agent, user_input, mode)
        print(c(f"\n  🤖 Agent: ", BOLD) + answer)
        print(c(f"  {'─'*55}\n", DIM))


if __name__ == "__main__":
    main()
