"""
main.py — Lab 3 Entry Point

3-WAY COMPARISON: Baseline Chatbot vs Agent v1 vs Agent v2 (RobustReActAgent)

Usage:
    python main.py              # full comparison (all 3 modes)
    python main.py --chatbot    # chatbot only
    python main.py --agent      # agent v1 only
    python main.py --v2         # agent v2 only

Demonstrates:
  - Why chatbots fail at multi-step e-commerce queries (no tools, hallucination)
  - How Agent v1 (ReAct loop) solves them with real tool calls
  - How Agent v2 (Guardrails + Retry) is more robust and production-ready
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv

# ── ensure project root is on sys.path ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# ── LLM Provider factory ────────────────────────────────────────────────────
def build_provider():
    """Instantiate the correct LLMProvider based on .env settings."""
    provider_name = os.getenv("DEFAULT_PROVIDER", "openai").lower()
    model_name = os.getenv("DEFAULT_MODEL", "gpt-4o")

    if provider_name == "openai":
        from src.core.openai_provider import OpenAIProvider
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"[Provider] OpenAI — model: {model_name}")
        return OpenAIProvider(model_name=model_name, api_key=api_key)

    elif provider_name == "google":
        from src.core.gemini_provider import GeminiProvider
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
        print(f"[Provider] Google Gemini — model: {model_name}")
        return GeminiProvider(model_name=model_name, api_key=api_key)

    elif provider_name == "local":
        from src.core.local_provider import LocalProvider
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        print(f"[Provider] Local CPU — model: {model_path}")
        return LocalProvider(model_path=model_path)

    else:
        raise ValueError(f"Unknown DEFAULT_PROVIDER: '{provider_name}'. Choose openai | google | local.")


# ── Test cases ───────────────────────────────────────────────────────────────
# Same test cases run against ALL 3 modes for fair comparison.
# Designed to expose chatbot limitations (multi-step, real data needed).
TEST_CASES = [
    # Simple — both should do OK
    {
        "id": "TC-01",
        "difficulty": "Simple",
        "question": "Is the iPhone 15 in stock?",
    },
    # Medium — needs price + stock
    {
        "id": "TC-02",
        "difficulty": "Medium",
        "question": "How much does the MacBook Pro cost and how many are available?",
    },
    # Hard — needs price + discount + shipping + calculation
    {
        "id": "TC-03",
        "difficulty": "Hard (multi-step)",
        "question": (
            "I want to buy 2 iPhone 15s using coupon code WINNER "
            "and ship to Hanoi. What is the total cost?"
        ),
    },
    # Hard — tests error handling (invalid coupon)
    {
        "id": "TC-04",
        "difficulty": "Hard (error case)",
        "question": (
            "I have coupon FAKE99. Can I apply it to buy 1 AirPods Pro "
            "with shipping to HCM?"
        ),
    },
]


def _separator(char="─", width=80):
    print(char * width)


def run_comparison(run_chatbot: bool = True, run_v1: bool = True, run_v2: bool = True):
    """
    3-way comparison: Baseline Chatbot vs Agent v1 vs Agent v2.

    Agent v1 = ReActAgent        (basic ReAct loop)
    Agent v2 = RobustReActAgent  (v1 + InputGuardrail + RetryHandler + OutputGuardrail)
    """
    llm = build_provider()

    from src.chatbot import SimpleChatbot
    from src.agent.agent import ReActAgent
    from src.agent.guardrails import RobustReActAgent   # ← Agent v2
    from src.tools import TOOLS_REGISTRY

    chatbot  = SimpleChatbot(llm=llm)
    agent_v1 = ReActAgent(llm=llm, tools=TOOLS_REGISTRY, max_steps=6)
    agent_v2 = RobustReActAgent(llm=llm, tools=TOOLS_REGISTRY, max_steps=6, max_retries=3)

    results = []

    for tc in TEST_CASES:
        _separator("═")
        print(f"🧪 {tc['id']} | {tc['difficulty']}")
        print(f"Question: {tc['question']}")
        _separator()

        chatbot_ms = v1_ms = v2_ms = None

        # ── Mode 1: Baseline Chatbot ─────────────────────────────────────────
        if run_chatbot:
            print("\n📢 [CHATBOT — no tools] Responding …")
            t0 = time.time()
            chatbot_answer = chatbot.chat(tc["question"])
            chatbot_ms = int((time.time() - t0) * 1000)
            print(f"Answer ({chatbot_ms} ms):\n{chatbot_answer}\n")

        # ── Mode 2: Agent v1 — basic ReAct ──────────────────────────────────
        if run_v1:
            print("🤖 [AGENT v1 — ReAct loop] Reasoning …")
            t0 = time.time()
            v1_answer = agent_v1.run(tc["question"])
            v1_ms = int((time.time() - t0) * 1000)
            print(f"\nAgent v1 Final Answer ({v1_ms} ms):\n{v1_answer}\n")

        # ── Mode 3: Agent v2 — Robust (retry + guardrails) ──────────────────
        if run_v2:
            print("🛡️  [AGENT v2 — Guardrails + Retry] Reasoning …")
            t0 = time.time()
            v2_answer = agent_v2.run(tc["question"])
            v2_ms = int((time.time() - t0) * 1000)
            print(f"\nAgent v2 Final Answer ({v2_ms} ms):\n{v2_answer}\n")

        results.append({
            "id": tc["id"],
            "difficulty": tc["difficulty"],
            "chatbot_ms": chatbot_ms,
            "v1_ms": v1_ms,
            "v2_ms": v2_ms,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    _separator("═")
    print("📊 PERFORMANCE SUMMARY — Chatbot vs Agent v1 vs Agent v2")
    _separator()
    header = f"{'ID':<8} {'Difficulty':<22} {'Chatbot ms':>12} {'Agent v1 ms':>12} {'Agent v2 ms':>12}"
    print(header)
    _separator("-")
    for r in results:
        c  = f"{r['chatbot_ms']:>12}" if r['chatbot_ms'] is not None else f"{'skip':>12}"
        v1 = f"{r['v1_ms']:>12}"      if r['v1_ms']      is not None else f"{'skip':>12}"
        v2 = f"{r['v2_ms']:>12}"      if r['v2_ms']      is not None else f"{'skip':>12}"
        row = f"{r['id']:<8} {r['difficulty']:<22} {c} {v1} {v2}"
        print(row)
    _separator("═")
    print("Logs saved to ./logs/ — inspect LLM_METRIC events for token usage & cost.")
    print("Agent v2 adds: InputGuardrail + RetryHandler(3x backoff) + OutputGuardrail")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 3: Chatbot vs Agent comparison")
    parser.add_argument("--chatbot", action="store_true", help="Run chatbot only")
    parser.add_argument("--agent",   action="store_true", help="Run Agent v1 only")
    parser.add_argument("--v2",      action="store_true", help="Run Agent v2 only")
    args = parser.parse_args()

    # If no flags given → run all 3
    run_all = not (args.chatbot or args.agent or args.v2)
    run_comparison(
        run_chatbot = run_all or args.chatbot,
        run_v1      = run_all or args.agent,
        run_v2      = run_all or args.v2,
    )
