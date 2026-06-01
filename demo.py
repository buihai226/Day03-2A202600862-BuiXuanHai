"""
demo.py — Interactive Agent Demo with Step-by-Step Tracing

Hiển thị rõ từng bước của ReAct loop:
  Thought → Action → Observation → ... → Final Answer

Chạy:  python demo.py
"""

import os
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# ── Colors ─────────────────────────────────────────────────────────────────
CYAN    = "\033[96m"
YELLOW  = "\033[93m"
GREEN   = "\033[92m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
DIM     = "\033[2m"

def c(text, color): return f"{color}{text}{RESET}"
def header(title): print(f"\n{BOLD}{'═'*65}{RESET}\n{BOLD}  {title}{RESET}\n{'═'*65}")
def sep(): print(c("─" * 65, DIM))


# ── Q&A bộ test: câu hỏi + đáp án mong đợi ──────────────────────────────
TEST_CASES = [
    {
        "id": "TC-01",
        "label": "Simple — Kiểm tra tồn kho",
        "question": "Is the iPhone 15 in stock?",
        "expected_answer": "10 units available",
        "expected_tools": ["check_stock"],
        "explain": (
            "Agent chỉ cần gọi 1 tool: check_stock('iPhone 15')\n"
            "  → DB trả về: 10 units\n"
            "  → Final Answer ngay sau 1 bước"
        ),
    },
    {
        "id": "TC-02",
        "label": "Medium — Giá + tồn kho",
        "question": "How much does the MacBook Pro cost and how many are left in stock?",
        "expected_answer": "45,000,000 VND — 3 units",
        "expected_tools": ["get_product_price", "check_stock"],
        "explain": (
            "Agent cần 2 tool calls:\n"
            "  Step 1: get_product_price('MacBook Pro') → 45,000,000 VND\n"
            "  Step 2: check_stock('MacBook Pro')       → 3 units\n"
            "  → Final Answer sau 2 bước"
        ),
    },
    {
        "id": "TC-03",
        "label": "Hard — Multi-step: mua + coupon + ship",
        "question": (
            "I want to buy 2 iPhone 15s using coupon WINNER "
            "and ship to Hanoi. What is the total cost?"
        ),
        "expected_answer": "45,030,000 VND",
        "expected_tools": [
            "get_product_price", "check_stock",
            "get_discount", "calc_shipping", "calculate_total"
        ],
        "explain": (
            "Agent phải chain 4-5 tool calls:\n"
            "  Step 1: get_product_price('iPhone 15') → 25,000,000 VND\n"
            "  Step 2: check_stock('iPhone 15')       → 10 units (đủ)\n"
            "  Step 3: get_discount('WINNER')          → 10% off\n"
            "  Step 4: calc_shipping(0.5, 'hanoi')    → 30,000 VND\n"
            "  Final:  2×25M - 10% + 30k = 45,030,000 VND"
        ),
    },
    {
        "id": "TC-04",
        "label": "Hard — Error recovery: coupon không hợp lệ",
        "question": (
            "I have coupon FAKE99. Can I use it to buy 1 AirPods Pro "
            "shipped to HCM?"
        ),
        "expected_answer": "FAKE99 invalid — total ~7,050,000 VND without discount",
        "expected_tools": ["get_discount", "check_stock", "get_product_price"],
        "explain": (
            "Agent gặp lỗi ở bước đầu:\n"
            "  Step 1: get_discount('FAKE99') → 'invalid or expired'\n"
            "  Agent KHÔNG dừng lại — tự điều chỉnh kế hoạch!\n"
            "  Step 2: check_stock('AirPods Pro') → 15 units\n"
            "  Step 3: get_product_price('AirPods Pro') → 7,000,000 VND\n"
            "  Final:  7M + 50k ship (HCM) = 7,050,000 VND (không giảm giá)"
        ),
    },
    {
        "id": "TC-05",
        "label": "Bonus — Discovery: tìm sản phẩm theo ngân sách",
        "question": "What smartphones can I buy for under 20 million VND?",
        "expected_answer": "iPhone 14 (18M) — Samsung S24 out of stock",
        "expected_tools": ["find_products_by_budget"],
        "explain": (
            "Dùng DISCOVERY tool (recommendation_tools.py):\n"
            "  find_products_by_budget(max_price_vnd=20000000, category=smartphone)\n"
            "  → iPhone 14: 18M ✅ còn hàng\n"
            "  → Samsung S24: 22M ❌ quá ngân sách\n"
            "  → iPhone 15: 25M ❌ quá ngân sách"
        ),
    },
]


# ── Verbose Agent wrapper — in từng bước ─────────────────────────────────
def run_verbose_agent(agent, question: str):
    """
    Chạy agent và in rõ từng bước Thought/Action/Observation.
    Không thay đổi code agent.py — chỉ đọc log sau.
    """
    from src.agent.agent import _parse_action, _parse_final_answer, _parse_kwargs

    # Tái hiện ReAct loop với output màu
    conversation = f"User Question: {question}\n\n"
    system_prompt = agent.get_system_prompt()

    step = 0
    start_total = time.time()

    while step < agent.max_steps:
        step += 1
        print(c(f"\n  ▶ Step {step}", YELLOW))

        t0 = time.time()
        result = agent.llm.generate(prompt=conversation, system_prompt=system_prompt)
        ms = int((time.time() - t0) * 1000)
        llm_output = result.get("content", "")
        tokens = result.get("usage", {}).get("total_tokens", "?")

        conversation += llm_output + "\n"

        # Parse và hiện Thought
        lines = llm_output.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.lower().startswith("thought:"):
                print(c(f"     💭 {line}", CYAN))
            elif line.lower().startswith("action:"):
                print(c(f"     ⚡ {line}", MAGENTA))
            elif line.lower().startswith("final answer:"):
                print(c(f"     ✅ {line}", GREEN))

        print(c(f"     {DIM}[{ms}ms | {tokens} tokens]{RESET}", DIM))

        # Check Final Answer
        final = _parse_final_answer(llm_output)
        if final:
            total_ms = int((time.time() - start_total) * 1000)
            return final, step, total_ms

        # Execute tool
        action = _parse_action(llm_output)
        if action:
            tool_name, raw_args = action
            observation = agent._execute_tool(tool_name, raw_args)
            print(c(f"     🔧 Tool: {tool_name}({raw_args})", MAGENTA))
            print(c(f"     📋 Obs: {observation[:100]}", CYAN))
            conversation += f"Observation: {observation}\n\n"
        else:
            conversation += "Observation: No action detected. Please call a tool or write Final Answer.\n\n"

    total_ms = int((time.time() - start_total) * 1000)
    return "Could not complete within max_steps.", step, total_ms


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    from src.core.openai_provider import OpenAIProvider
    from src.agent.agent import ReActAgent
    from src.tools import TOOLS_REGISTRY

    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("DEFAULT_MODEL", "gpt-4o")

    print(c(f"\n{'='*65}", BOLD))
    print(c(f"  🤖 ReAct Agent Demo — {model}", BOLD))
    print(c(f"  Tools available: {len(TOOLS_REGISTRY)} tools", BOLD))
    print(c(f"{'='*65}\n", BOLD))
    print(c("  📚 Đây là CSDL mock của Agent:\n", DIM))
    print(c("  Sản phẩm:", DIM))
    print("    iPhone 15     → 25,000,000 VND | stock: 10")
    print("    iPhone 14     → 18,000,000 VND | stock: 5")
    print("    Samsung S24   → 22,000,000 VND | stock: 0 (hết)")
    print("    MacBook Pro   → 45,000,000 VND | stock: 3")
    print("    AirPods Pro   →  7,000,000 VND | stock: 15")
    print(c("\n  Coupons:", DIM))
    print("    WINNER  → 10% off  |  SALE20 → 20%  |  VIP50 → 50%  |  NEWUSER → 15%")
    print(c("\n  Shipping:", DIM))
    print("    Hanoi: 30,000 VND  |  HCM: 50,000 VND  |  Đà Nẵng: 40,000 VND")

    llm   = OpenAIProvider(model_name=model, api_key=api_key)
    agent = ReActAgent(llm=llm, tools=TOOLS_REGISTRY, max_steps=6)

    results = []
    for tc in TEST_CASES:
        header(f"{tc['id']} — {tc['label']}")

        print(c("  📖 LUỒNG HOẠT ĐỘNG DỰ KIẾN:", YELLOW))
        for line in tc["explain"].split("\n"):
            print(c(f"  {line}", DIM))

        print(c(f"\n  ❓ Câu hỏi:", BOLD))
        print(f"  {tc['question']}")
        print(c(f"\n  🎯 Đáp án mong đợi: {tc['expected_answer']}", GREEN))
        print(c(f"\n  🔧 Tools cần dùng: {' → '.join(tc['expected_tools'])}", MAGENTA))

        sep()
        print(c("  ⚙️  AGENT ĐANG CHẠY...\n", YELLOW))

        answer, steps, total_ms = run_verbose_agent(agent, tc["question"])

        sep()
        print(c(f"\n  📬 Final Answer ({steps} bước | {total_ms}ms):", BOLD))
        print(f"  {answer}")

        # Kiểm tra kết quả
        expected_tokens = tc["expected_answer"].split(" — ")[0].replace(",", "").lower()
        actual_lower = answer.lower().replace(",", "")
        correct = any(tok.replace(",","").lower() in actual_lower
                      for tok in tc["expected_answer"].split() if len(tok) > 3)

        status = c("✅ ĐÚNG", GREEN) if correct else c("⚠️  KIỂM TRA LẠI", YELLOW)
        print(c(f"\n  {status}", BOLD))
        results.append({"id": tc["id"], "steps": steps, "ms": total_ms, "ok": correct})

        print()
        input(c("  [Enter để chạy câu tiếp theo...]", DIM))

    # ── Summary ───────────────────────────────────────────────────────────
    header("📊 KẾT QUẢ TỔNG KẾT")
    print(f"  {'ID':<8} {'Label':<35} {'Steps':>6} {'Time':>8} {'OK?':>6}")
    print("  " + "─" * 58)
    for r in results:
        ok_str = c("✅", GREEN) if r["ok"] else c("⚠️", YELLOW)
        label  = next(tc["label"] for tc in TEST_CASES if tc["id"] == r["id"])
        print(f"  {r['id']:<8} {label[:34]:<35} {r['steps']:>6} {r['ms']:>6}ms {ok_str:>6}")

    total_ok = sum(1 for r in results if r["ok"])
    print(f"\n  Score: {total_ok}/{len(results)} đúng")
    print(c("\n  💡 Key insight:", BOLD))
    print("  Agent gọi tool từng bước — mỗi Observation cung cấp thông tin")
    print("  để LLM quyết định bước tiếp theo. Đây là điểm khác biệt cốt")
    print("  lõi so với Chatbot (chỉ 1 LLM call, không có tool nào).")
    print()


if __name__ == "__main__":
    main()
