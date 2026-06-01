"""
src/chatbot.py

Chatbot Baseline — Phase 2 of Lab 3.

PURPOSE:
    Demonstrate the fundamental limitations of a standard LLM chatbot
    when faced with real-world e-commerce queries that require:
      - Real-time data (stock levels, current prices)
      - Multi-step calculations (discount + shipping + total)
      - Accessing external systems (inventory DB, coupon DB)

    The chatbot has NO tools, NO database access, NO calculation ability.
    It can only answer based on what the LLM was trained on — which means
    it will hallucinate prices, make up stock levels, and produce wrong totals.

HOW IT WORKS:
    User Input → LLM (single generate() call) → Raw LLM Response

    No ReAct loop. No tool execution. No Observation feedback.

COMPARISON TARGET:
    Run this chatbot on the same test cases as the ReAct Agent (agent.py)
    to clearly see WHY agents are needed for real-world tasks.

FEATURES:
    1. SimpleChatbot      — single-turn, one question = one answer
    2. ConversationalChatbot — multi-turn with conversation memory
"""

from typing import Optional, List, Dict
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt — defines chatbot personality and behavior
# ─────────────────────────────────────────────────────────────────────────────
CHATBOT_SYSTEM_PROMPT = """You are a helpful e-commerce customer support assistant for a Vietnamese online store.

Your role:
- Answer customer questions about products, pricing, orders, and shipping
- Be friendly, concise, and professional
- If you don't know exact real-time data (stock levels, current promotions),
  make your best general estimate and CLEARLY state it is an estimate

Limitations you must be honest about:
- You do NOT have access to live inventory systems
- You do NOT have real-time pricing data
- You CANNOT look up specific coupon codes
- You CANNOT calculate exact shipping costs without a logistics API

Always try to be helpful even within your limitations.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. SimpleChatbot — Single-turn baseline
#    One question in → one answer out. No memory between calls.
#    This is the most basic LLM usage pattern.
# ─────────────────────────────────────────────────────────────────────────────
class SimpleChatbot:
    """
    Baseline chatbot: a single LLM call per user message.

    Limitations (compared to ReAct Agent):
      - No tools → cannot access real stock/price/coupon data
      - No memory → each question is independent
      - No multi-step reasoning → cannot chain calculations
      - Relies purely on LLM training data → may hallucinate facts

    Use this as the BASELINE to compare against the ReAct Agent.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def chat(self, user_input: str) -> str:
        """
        Send one user message to the LLM and return the raw reply.

        Flow:
            user_input → LLM.generate() → response text

        No tools. No loops. No observations.
        """
        logger.log_event("CHATBOT_START", {
            "mode": "simple",
            "input": user_input,
            "model": self.llm.model_name,
        })

        # Single LLM call — this is the KEY difference from the ReAct Agent
        result = self.llm.generate(
            prompt=user_input,
            system_prompt=CHATBOT_SYSTEM_PROMPT,
        )

        response_text: str = result.get("content", "")
        usage: Dict = result.get("usage", {})
        latency_ms: int = result.get("latency_ms", 0)

        # Track telemetry (same as Agent for fair comparison)
        tracker.track_request(
            provider=result.get("provider", "unknown"),
            model=self.llm.model_name,
            usage=usage,
            latency_ms=latency_ms,
        )

        logger.log_event("CHATBOT_RESPONSE", {
            "mode": "simple",
            "latency_ms": latency_ms,
            "total_tokens": usage.get("total_tokens", 0),
            "response_preview": response_text[:300],
        })

        return response_text


# ─────────────────────────────────────────────────────────────────────────────
# 2. ConversationalChatbot — Multi-turn with memory
#    Maintains conversation history so the LLM can reference prior messages.
#    Still NO tools — but can sustain a dialogue.
# ─────────────────────────────────────────────────────────────────────────────
class ConversationalChatbot:
    """
    Multi-turn chatbot that accumulates conversation history.

    Unlike SimpleChatbot, this maintains a growing conversation buffer
    so the LLM can reference previous exchanges within the same session.

    Still LIMITED by:
      - No tools → cannot fetch real data
      - Context window → history eventually gets too long and must be trimmed
      - Still hallucination-prone for factual queries

    This better approximates a "real chatbot" experience, while still
    demonstrating why tools (ReAct Agent) are necessary for accuracy.
    """

    def __init__(self, llm: LLMProvider, max_history_turns: int = 10):
        self.llm = llm
        self.max_history_turns = max_history_turns
        # History stored as list of (role, content) tuples
        self.history: List[Dict[str, str]] = []
        self.turn_count: int = 0

    def chat(self, user_input: str) -> str:
        """
        Send user message with full conversation history to LLM.

        The growing conversation is injected into the prompt so the LLM
        can maintain context across multiple turns.
        """
        self.turn_count += 1

        logger.log_event("CHATBOT_START", {
            "mode": "conversational",
            "turn": self.turn_count,
            "input": user_input,
            "history_length": len(self.history),
            "model": self.llm.model_name,
        })

        # Add user message to history
        self.history.append({"role": "user", "content": user_input})

        # Trim history if too long (keep last N turns)
        if len(self.history) > self.max_history_turns * 2:
            # Keep system context and recent turns
            self.history = self.history[-(self.max_history_turns * 2):]

        # Build prompt with conversation history embedded
        conversation_text = self._build_conversation_prompt()

        # Single LLM call — same as SimpleChatbot, just with more context
        result = self.llm.generate(
            prompt=conversation_text,
            system_prompt=CHATBOT_SYSTEM_PROMPT,
        )

        response_text: str = result.get("content", "")
        usage: Dict = result.get("usage", {})
        latency_ms: int = result.get("latency_ms", 0)

        # Add assistant reply to history for next turn
        self.history.append({"role": "assistant", "content": response_text})

        tracker.track_request(
            provider=result.get("provider", "unknown"),
            model=self.llm.model_name,
            usage=usage,
            latency_ms=latency_ms,
        )

        logger.log_event("CHATBOT_RESPONSE", {
            "mode": "conversational",
            "turn": self.turn_count,
            "latency_ms": latency_ms,
            "total_tokens": usage.get("total_tokens", 0),
            "response_preview": response_text[:300],
        })

        return response_text

    def _build_conversation_prompt(self) -> str:
        """
        Format conversation history as a single prompt string.
        The LLM sees the full exchange and can reference earlier messages.
        """
        lines = []
        for msg in self.history:
            role_label = "Customer" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")
        # The last line is the current user turn — LLM should respond to it
        return "\n\n".join(lines) + "\n\nAssistant:"

    def reset(self):
        """Clear conversation history for a new session."""
        self.history.clear()
        self.turn_count = 0
        logger.log_event("CHATBOT_RESET", {"mode": "conversational"})


# ─────────────────────────────────────────────────────────────────────────────
# Demo / quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick demo: run both chatbot modes against a sample e-commerce question.
    Requires DEFAULT_PROVIDER and API keys set in .env
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from dotenv import load_dotenv
    load_dotenv()

    from src.core.openai_provider import OpenAIProvider

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("DEFAULT_MODEL", "gpt-4o")
    llm = OpenAIProvider(model_name=model, api_key=api_key)

    test_question = (
        "I want to buy 2 iPhone 15s using coupon WINNER and ship to Hanoi. "
        "What is the total cost?"
    )

    print("=" * 60)
    print("SIMPLE CHATBOT (no memory, no tools)")
    print("=" * 60)
    simple = SimpleChatbot(llm)
    print(f"Q: {test_question}")
    print(f"A: {simple.chat(test_question)}\n")

    print("=" * 60)
    print("CONVERSATIONAL CHATBOT (memory, no tools)")
    print("=" * 60)
    convo = ConversationalChatbot(llm)
    q1 = "What iPhone models do you have?"
    q2 = "How much does the iPhone 15 cost?"
    q3 = test_question

    for q in [q1, q2, q3]:
        print(f"Q: {q}")
        print(f"A: {convo.chat(q)}\n")
