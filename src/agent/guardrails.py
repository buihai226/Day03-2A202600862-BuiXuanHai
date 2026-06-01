"""
src/agent/guardrails.py

Guardrails & Retry Logic — Failure Handling Bonus (+3 points)

Provides:
  1. InputGuardrail  — validate/sanitize user input before sending to Agent
  2. RetryHandler    — wrap LLM calls with exponential backoff retry
  3. OutputGuardrail — validate final answer is coherent and not empty
  4. RobustReActAgent — drop-in replacement for ReActAgent with all guardrails
"""

import re
import time
import random
from typing import List, Dict, Any, Optional, Callable
from src.telemetry.logger import logger


# ────────────────────────────────────────────────────────────────────────────
# 1. Input Guardrails
# ────────────────────────────────────────────────────────────────────────────

class InputGuardrail:
    """
    Validates and sanitizes user input before it reaches the Agent.
    Catches prompt injection attempts, empty inputs, and oversized inputs.
    """

    MAX_INPUT_LENGTH = 500  # characters
    # Patterns that look like prompt injection
    INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"forget (everything|all) (you know|above)",
        r"you are now",
        r"act as (a |an )?(different|new|evil|unrestricted)",
        r"jailbreak",
        r"DAN mode",
    ]

    def __init__(self):
        self._compiled = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """
        Returns (is_valid, sanitized_input_or_error_message).
        """
        # Check empty
        stripped = user_input.strip()
        if not stripped:
            return False, "Input is empty. Please type a question."

        # Check length
        if len(stripped) > self.MAX_INPUT_LENGTH:
            return False, (
                f"Input too long ({len(stripped)} chars). "
                f"Please keep it under {self.MAX_INPUT_LENGTH} characters."
            )

        # Check prompt injection
        for pattern in self._compiled:
            if pattern.search(stripped):
                logger.log_event("GUARDRAIL_INJECTION_BLOCKED", {
                    "pattern": pattern.pattern,
                    "input_preview": stripped[:100],
                })
                return False, (
                    "I'm sorry, your message contains content I cannot process. "
                    "Please ask a genuine shopping question."
                )

        return True, stripped


# ────────────────────────────────────────────────────────────────────────────
# 2. Retry Handler
# ────────────────────────────────────────────────────────────────────────────

class RetryHandler:
    """
    Wraps any callable with exponential backoff + jitter retry logic.

    Default: 3 attempts, starting at 1s, doubling each time with ±30% jitter.
    Covers transient API errors (rate limits, network timeouts, 5xx).
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_s: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.3,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay_s
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Call fn(*args, **kwargs) with retry on exception.
        Raises the last exception if all retries are exhausted.
        """
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                if attempt > 1:
                    logger.log_event("RETRY_SUCCESS", {"attempt": attempt})
                return result

            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
                # Add jitter: delay × (1 ± jitter)
                jitter_factor = 1 + self.jitter * (2 * random.random() - 1)
                sleep_time = delay * jitter_factor

                logger.log_event("RETRY_ATTEMPT", {
                    "attempt": attempt,
                    "max_retries": self.max_retries,
                    "error": str(exc)[:200],
                    "sleep_s": round(sleep_time, 2),
                })

                if attempt < self.max_retries:
                    time.sleep(sleep_time)

        logger.log_event("RETRY_EXHAUSTED", {
            "max_retries": self.max_retries,
            "final_error": str(last_exception)[:200],
        })
        raise last_exception


# ────────────────────────────────────────────────────────────────────────────
# 3. Output Guardrails
# ────────────────────────────────────────────────────────────────────────────

class OutputGuardrail:
    """
    Validates the final answer produced by the Agent before returning to user.
    Catches empty answers, hallucinated prices, and suspiciously short responses.
    """

    MIN_ANSWER_LENGTH = 10   # chars
    # If answer contains obviously wrong patterns, flag it
    SUSPICIOUS_PATTERNS = [
        r"\$\d+",           # USD price in a VND system
        r"as of (my |the )?training",   # LLM falling back to training knowledge
        r"I (don't|do not|cannot) have (access|real-time)",
    ]

    def __init__(self):
        self._compiled = [
            re.compile(p, re.IGNORECASE) for p in self.SUSPICIOUS_PATTERNS
        ]

    def validate(self, answer: str) -> tuple[bool, str, List[str]]:
        """
        Returns (is_valid, answer, list_of_warnings).
        Warnings are logged but do NOT block the answer.
        """
        warnings: List[str] = []

        if not answer or len(answer.strip()) < self.MIN_ANSWER_LENGTH:
            return False, "I was unable to generate a response. Please try again.", []

        for pattern in self._compiled:
            if pattern.search(answer):
                warnings.append(f"Suspicious pattern detected: '{pattern.pattern}'")

        if warnings:
            logger.log_event("OUTPUT_GUARDRAIL_WARNING", {
                "warnings": warnings,
                "answer_preview": answer[:200],
            })

        return True, answer.strip(), warnings


# ────────────────────────────────────────────────────────────────────────────
# 4. RobustReActAgent — Agent v2 with all guardrails + retry
# ────────────────────────────────────────────────────────────────────────────

from src.agent.agent import ReActAgent  # Import base agent
from src.core.llm_provider import LLMProvider


class RobustReActAgent(ReActAgent):
    """
    Agent v2: ReActAgent wrapped with:
      - InputGuardrail  (before run)
      - RetryHandler    (around each LLM call)
      - OutputGuardrail (before returning Final Answer)

    Inherits all ReAct loop logic from ReActAgent.
    Only overrides run() to inject the guardrail/retry layer.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: List[Dict[str, Any]],
        max_steps: int = 6,
        max_retries: int = 3,
    ):
        super().__init__(llm=llm, tools=tools, max_steps=max_steps)
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()
        self.retry = RetryHandler(max_retries=max_retries)

    def run(self, user_input: str) -> str:
        """
        v2 run() with guardrails + retry.
        Follows the same ReAct loop but:
          1. Validates input first
          2. Wraps LLM generate() with retry
          3. Validates output before returning
        """
        # ── Input Guardrail ────────────────────────────────────────────────
        is_valid, sanitized = self.input_guard.validate(user_input)
        if not is_valid:
            logger.log_event("GUARDRAIL_INPUT_REJECTED", {"reason": sanitized})
            return sanitized

        logger.log_event("AGENT_V2_START", {
            "input": sanitized,
            "model": self.llm.model_name,
            "max_steps": self.max_steps,
            "max_retries": self.retry.max_retries,
        })

        # ── ReAct Loop (same logic, but LLM call wrapped in retry) ─────────
        from src.agent.agent import (
            _parse_final_answer,
            _parse_action,
        )
        from src.telemetry.metrics import tracker

        conversation = f"User Question: {sanitized}\n\n"
        final_answer: Optional[str] = None

        for step in range(self.max_steps):
            logger.log_event("AGENT_V2_STEP_START", {"step": step + 1})

            # LLM call WITH retry
            try:
                result = self.retry.call(
                    self.llm.generate,
                    prompt=conversation,
                    system_prompt=self.get_system_prompt(),
                )
            except Exception as exc:
                logger.log_event("AGENT_V2_LLM_FAILED", {"error": str(exc)})
                return (
                    "Sorry, I encountered a technical issue after multiple retries. "
                    "Please try again later."
                )

            llm_output: str = result.get("content", "")
            usage = result.get("usage", {})
            latency_ms = result.get("latency_ms", 0)

            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=usage,
                latency_ms=latency_ms,
            )

            conversation += llm_output + "\n"

            # Check Final Answer
            final_answer = _parse_final_answer(llm_output)
            if final_answer:
                logger.log_event("AGENT_V2_FINAL_ANSWER", {
                    "step": step + 1,
                    "answer_preview": final_answer[:200],
                })
                break

            # Parse & execute tool
            action = _parse_action(llm_output)
            if action is None:
                conversation += (
                    "Observation: No action detected. "
                    "Please call a tool or write 'Final Answer:'.\n\n"
                )
                continue

            tool_name, raw_args = action
            observation = self._execute_tool(tool_name, raw_args)

            logger.log_event("AGENT_V2_OBSERVATION", {
                "step": step + 1,
                "tool": tool_name,
                "observation": observation,
            })

            conversation += f"Observation: {observation}\n\n"

        else:
            logger.log_event("AGENT_V2_TIMEOUT", {"max_steps": self.max_steps})
            final_answer = (
                f"I could not complete the task within {self.max_steps} steps. "
                "Please try a simpler query."
            )

        # ── Output Guardrail ───────────────────────────────────────────────
        is_valid, validated_answer, warnings = self.output_guard.validate(
            final_answer or ""
        )

        if not is_valid:
            return validated_answer  # error message

        if warnings:
            # Append a soft disclaimer so user knows something was flagged
            validated_answer += (
                "\n\n_(Note: Some parts of this answer may be estimates. "
                "Please verify with the product page for the latest info.)_"
            )

        return validated_answer
