"""
src/agent/agent.py

ReAct Agent — Reasoning + Acting loop.

The agent follows the classic Thought → Action → Observation cycle:
  1. LLM generates a "Thought" (reasoning) + "Action" (tool call).
  2. We parse the Action string to find the tool name and arguments.
  3. We execute the tool and feed the result back as an "Observation".
  4. Steps repeat until the LLM emits "Final Answer:" or max_steps is reached.

Key design decisions:
  - Conversation history is accumulated as a single growing prompt string so
    older Observations remain visible to the LLM on every turn.
  - Parsing is intentionally lenient: we accept both `tool(arg)` and
    `tool(key=val, ...)` formats and strip common LLM markdown artefacts.
  - max_steps acts as the hard circuit-breaker to prevent infinite billing.
"""

import re
import time
from typing import List, Dict, Any, Optional

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


# ──────────────────────────────────────────────────────────────────────────────
# Regex helpers
# ──────────────────────────────────────────────────────────────────────────────

# Matches:  Action: tool_name(anything here)
_ACTION_RE = re.compile(
    r"Action\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)",
    re.IGNORECASE,
)

# Matches:  Final Answer: (rest of the line / block)
_FINAL_RE = re.compile(
    r"Final\s+Answer\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_action(text: str) -> Optional[tuple[str, str]]:
    """
    Extract (tool_name, raw_args_string) from an LLM output block.
    Returns None if no Action line is found.
    """
    m = _ACTION_RE.search(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _parse_final_answer(text: str) -> Optional[str]:
    """Return the Final Answer text if present, else None."""
    m = _FINAL_RE.search(text)
    if m:
        # Only return the first line/block, strip excess whitespace
        return m.group(1).strip()
    return None


def _parse_kwargs(raw_args: str) -> Dict[str, str]:
    """
    Convert a raw argument string into a dict.

    Supports formats:
      'iPhone 15'                         → {'_arg0': 'iPhone 15'}
      'weight_kg=0.5, destination=hanoi'  → {'weight_kg': '0.5', 'destination': 'hanoi'}
      'unit_price=25000000, quantity=2'   → {'unit_price': '25000000', 'quantity': '2'}
    """
    raw_args = raw_args.strip().strip("'\"")
    if not raw_args:
        return {}

    result: Dict[str, str] = {}

    # Try key=value pairs first
    kv_pairs = re.findall(r"(\w+)\s*=\s*([^,]+)", raw_args)
    if kv_pairs:
        for k, v in kv_pairs:
            result[k.strip()] = v.strip().strip("'\"")
        return result

    # Fallback: treat the whole string as a single positional argument
    result["_arg0"] = raw_args.strip().strip("'\"")
    return result


def _call_tool_with_kwargs(fn, kwargs: Dict[str, str]) -> str:
    """
    Call a tool function using parsed kwargs.
    Falls back to positional arg if kwargs contains only '_arg0'.
    """
    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    if "_arg0" in kwargs and len(kwargs) == 1:
        # Single positional argument — pass to the first parameter
        value = kwargs["_arg0"]
        try:
            # Attempt numeric coercion so calc_shipping gets a float
            return fn(float(value))
        except (ValueError, TypeError):
            return fn(value)

    # Named kwargs — coerce types by inspecting annotations
    typed_kwargs: Dict[str, Any] = {}
    annotations = fn.__annotations__

    for k, v in kwargs.items():
        if k not in params:
            continue  # Skip unknown parameters
        ann = annotations.get(k, str)
        try:
            if ann in (int, float) or ann == "float" or ann == "int":
                typed_kwargs[k] = ann(v)
            else:
                typed_kwargs[k] = v
        except (ValueError, TypeError):
            typed_kwargs[k] = v  # Pass as string if coercion fails

    return fn(**typed_kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# ReAct Agent
# ──────────────────────────────────────────────────────────────────────────────

class ReActAgent:
    """
    A ReAct-style Agent that follows the Thought → Action → Observation loop.

    Args:
        llm       : Any LLMProvider (OpenAI / Gemini / Local).
        tools     : List of tool dicts with keys 'name', 'description', 'function'.
        max_steps : Hard limit on Thought→Action iterations (circuit-breaker).
    """

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 6):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        # Maintain conversation history as a list of message dicts for context
        self.history: List[Dict[str, str]] = []

    # ──────────────────────────────────────────────────────────────
    # System prompt
    # ──────────────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """
        Build the system prompt that instructs the LLM to follow ReAct format.

        Improvements over the skeleton:
          - Strict format enforcement with examples.
          - Explicit rule: stop as soon as you have enough info (avoid extra loops).
          - No markdown/code-blocks in Action lines (prevents JSON parse errors).
        """
        tool_descriptions = "\n".join(
            [f"  - {t['name']}: {t['description']}" for t in self.tools]
        )
        return f"""You are an intelligent E-commerce Assistant that helps users with shopping queries.
You have access to the following tools:

{tool_descriptions}

STRICT FORMAT RULES — follow these exactly, no markdown, no code blocks:

  Thought: <your reasoning about what to do next>
  Action: tool_name(arg1_name=arg1_value, arg2_name=arg2_value)
  Observation: <tool result will appear here — do NOT write this yourself>

Repeat Thought/Action until you have all the information you need, then write:

  Final Answer: <your complete, friendly response to the user>

IMPORTANT RULES:
1. Only call one tool per Action line.
2. Do NOT invent tool names — only use the tools listed above.
3. Do NOT write the Observation yourself — wait for it.
4. When you have enough information to answer, write "Final Answer:" immediately.
5. Never loop more than necessary; combine information in the Final Answer.
6. Always use named keyword arguments (key=value) in Action calls.

Example:
  Thought: I need to check the price of iPhone 15.
  Action: get_product_price(item_name=iPhone 15)
  Observation: Unit price of 'iPhone 15': 25,000,000 VND.
  Thought: Now I have the price. I can answer.
  Final Answer: The iPhone 15 costs 25,000,000 VND.
"""

    # ──────────────────────────────────────────────────────────────
    # Main ReAct loop
    # ──────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> str:
        """
        Execute the ReAct loop for a user query.

        Flow:
          1. Build the growing prompt (system_prompt + conversation so far).
          2. Call LLM → receive Thought + Action.
          3. Parse Action → execute tool → append Observation.
          4. If LLM emits "Final Answer:" → stop and return it.
          5. If max_steps exceeded → return timeout message.
        """
        logger.log_event("AGENT_START", {
            "input": user_input,
            "model": self.llm.model_name,
            "max_steps": self.max_steps,
        })

        # Accumulate the conversation as a single growing text block.
        # The LLM sees: previous Thought/Action/Observation + new user turn.
        conversation = f"User Question: {user_input}\n\n"

        final_answer = None

        for step in range(self.max_steps):          # ← hard loop limit
            logger.log_event("AGENT_STEP_START", {"step": step + 1})

            # ── 1. Generate next Thought + Action from LLM ──────────────────
            result = self.llm.generate(
                prompt=conversation,
                system_prompt=self.get_system_prompt(),
            )

            llm_output: str = result.get("content", "")
            usage: Dict = result.get("usage", {})
            latency_ms: int = result.get("latency_ms", 0)

            # Track telemetry
            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=usage,
                latency_ms=latency_ms,
            )

            logger.log_event("LLM_OUTPUT", {
                "step": step + 1,
                "output_preview": llm_output[:300],
            })

            # Append LLM output to the growing conversation
            conversation += llm_output + "\n"

            # ── 2. Check for Final Answer ────────────────────────────────────
            final_answer = _parse_final_answer(llm_output)
            if final_answer:
                logger.log_event("AGENT_FINAL_ANSWER", {
                    "step": step + 1,
                    "answer_preview": final_answer[:200],
                })
                break   # ← exit the loop cleanly

            # ── 3. Parse Action ──────────────────────────────────────────────
            action = _parse_action(llm_output)
            if action is None:
                # LLM produced no Action and no Final Answer — nudge it
                logger.log_event("AGENT_NO_ACTION", {"step": step + 1})
                conversation += "Observation: No action was detected. Please continue reasoning and call a tool or provide a Final Answer.\n\n"
                continue    # ← go to next iteration

            tool_name, raw_args = action
            logger.log_event("AGENT_TOOL_CALL", {
                "step": step + 1,
                "tool": tool_name,
                "raw_args": raw_args,
            })

            # ── 4. Execute tool → get Observation ───────────────────────────
            observation = self._execute_tool(tool_name, raw_args)

            logger.log_event("AGENT_OBSERVATION", {
                "step": step + 1,
                "tool": tool_name,
                "observation": observation,
            })

            # Append the Observation so LLM sees it on the next turn
            conversation += f"Observation: {observation}\n\n"

        else:
            # Loop exhausted without a Final Answer → timeout
            logger.log_event("AGENT_TIMEOUT", {"max_steps": self.max_steps})
            final_answer = (
                f"I'm sorry, I could not complete the task within {self.max_steps} steps. "
                "Please try rephrasing your question or ask a simpler query."
            )

        logger.log_event("AGENT_END", {
            "final_answer_preview": (final_answer or "")[:200],
        })

        return final_answer or "I was unable to produce a final answer."

    # ──────────────────────────────────────────────────────────────
    # Tool dispatcher
    # ──────────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, raw_args: str) -> str:
        """
        Locate the tool by name and call it with the parsed arguments.

        Returns the tool's string result (Observation) or an error message.
        """
        # Find tool in registry
        matched_tool = None
        for tool in self.tools:
            if tool["name"].lower() == tool_name.lower():
                matched_tool = tool
                break

        if matched_tool is None:
            available = ", ".join(t["name"] for t in self.tools)
            return (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {available}."
            )

        fn = matched_tool.get("function")
        if fn is None:
            return f"Error: Tool '{tool_name}' has no callable function registered."

        # Parse args and call
        try:
            kwargs = _parse_kwargs(raw_args)
            return _call_tool_with_kwargs(fn, kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Tool '{tool_name}' raised an exception: {exc}")
            return f"Error executing '{tool_name}': {exc}"
