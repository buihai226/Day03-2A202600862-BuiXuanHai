import time
from typing import Dict, Any, List
from src.telemetry.logger import logger

class PerformanceTracker:
    """
    Tracking industry-standard metrics for LLMs.
    """
    def __init__(self):
        self.session_metrics = []

    def track_request(self, provider: str, model: str, usage: Dict[str, int], latency_ms: int):
        """
        Logs a single request metric to our telemetry.
        """
        metric = {
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "cost_estimate": self._calculate_cost(model, usage) # Mock cost calculation
        }
        self.session_metrics.append(metric)
        logger.log_event("LLM_METRIC", metric)

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        Estimate cost in USD based on model pricing (per 1K tokens).
        Prices are approximate and may change — update as needed.
        """
        pricing = {
            # OpenAI
            "gpt-4o":           {"prompt": 0.005,  "completion": 0.015},
            "gpt-4o-mini":      {"prompt": 0.00015,"completion": 0.0006},
            "gpt-4-turbo":      {"prompt": 0.01,   "completion": 0.03},
            "gpt-3.5-turbo":    {"prompt": 0.0005, "completion": 0.0015},
            # Google Gemini
            "gemini-1.5-flash": {"prompt": 0.00035,"completion": 0.00105},
            "gemini-1.5-pro":   {"prompt": 0.0035, "completion": 0.0105},
            "gemini-pro":       {"prompt": 0.0005, "completion": 0.0015},
        }
        rates = pricing.get(model, {"prompt": 0.001, "completion": 0.002})
        prompt_cost = (usage.get("prompt_tokens", 0) / 1000) * rates["prompt"]
        completion_cost = (usage.get("completion_tokens", 0) / 1000) * rates["completion"]
        return round(prompt_cost + completion_cost, 6)

# Global tracker instance
tracker = PerformanceTracker()
