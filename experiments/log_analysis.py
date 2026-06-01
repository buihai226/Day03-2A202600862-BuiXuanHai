"""
experiments/log_analysis.py — Parse logs/ và tính Aggregate Reliability Metrics

Theo yêu cầu EVALUATION.md:
  1. Token Efficiency   — prompt vs completion, cost per task
  2. Latency           — P50, P99, per-step latency
  3. Loop Count        — steps per task, termination quality
  4. Failure Analysis  — parser errors, hallucinations, timeouts

Chạy: python experiments/log_analysis.py
      python experiments/log_analysis.py --log logs/2026-06-01.log
"""

import os, sys, json, glob, argparse
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Colors ──────────────────────────────────────────────────────
CYAN   = "\033[96m"; YELLOW = "\033[93m"; GREEN  = "\033[92m"
MAGENTA= "\033[95m"; RED    = "\033[91m"; BOLD   = "\033[1m"
RESET  = "\033[0m";  DIM    = "\033[2m"
def c(t, col): return f"{col}{t}{RESET}"
def header(t): print(f"\n{BOLD}{'═'*62}{RESET}\n{BOLD}  {t}{RESET}\n{'═'*62}")
def sub(t):    print(f"\n{c(t, YELLOW)}")


def load_logs(log_path: str) -> list:
    events = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def group_sessions(events: list) -> dict:
    """
    Nhóm events thành các session riêng biệt dựa vào:
      - CHATBOT_START → chatbot session
      - AGENT_START   → agent session (v1 hoặc v2)
    Trả về: {'chatbot': [...sessions], 'agent_v1': [...], 'agent_v2': [...]}
    """
    sessions = {"chatbot": [], "agent_v1": [], "agent_v2": []}
    current = None
    current_type = None

    for ev in events:
        etype = ev["event"]

        if etype == "CHATBOT_START":
            if current:
                sessions[current_type].append(current)
            current = [ev]
            current_type = "chatbot"

        elif etype == "AGENT_START":
            if current:
                sessions[current_type].append(current)
            # v2 nếu có guardrail events, v1 nếu không (xác định sau)
            current = [ev]
            current_type = "agent_v1"  # default, update later

        elif etype == "GUARDRAIL_INJECTION_BLOCKED":
            # Đây là event của agent v2
            if current_type and "agent" in (current_type or ""):
                current_type = "agent_v2"
            if current is not None:
                current.append(ev)

        elif current is not None:
            current.append(ev)
            # Detect v2 by guardrail events in session
            if etype in ("GUARDRAIL_INPUT_BLOCKED", "GUARDRAIL_OUTPUT_BLOCKED",
                         "RETRY_ATTEMPT"):
                current_type = "agent_v2"

    if current:
        sessions[current_type].append(current)

    return sessions


def analyze_chatbot_session(session: list) -> dict:
    """Extract metrics từ 1 chatbot session."""
    metrics = {"latency_ms": None, "total_tokens": 0, "cost": 0, "input": ""}
    for ev in session:
        if ev["event"] == "CHATBOT_START":
            metrics["input"] = ev["data"].get("input", "")
        elif ev["event"] == "CHATBOT_RESPONSE":
            d = ev["data"]
            metrics["latency_ms"]  = d.get("latency_ms", 0)
            metrics["total_tokens"] = d.get("tokens", {}).get("total_tokens", 0)
        elif ev["event"] == "LLM_METRIC":
            metrics["cost"] = ev["data"].get("cost_estimate", 0)
    return metrics


def analyze_agent_session(session: list) -> dict:
    """Extract metrics từ 1 agent session."""
    metrics = {
        "input": "", "steps": 0, "total_tokens": 0, "total_latency_ms": 0,
        "cost": 0, "tool_calls": [], "final_answer": False,
        "timeout": False, "step_latencies": [],
        "parser_errors": 0, "hallucinated_tools": 0,
    }
    # Known valid tool names
    VALID_TOOLS = {
        "check_stock","get_product_price","get_discount","calc_shipping","calculate_total",
        "find_products_by_budget","compare_products","get_active_promotions","find_alternative_product",
    }
    llm_metrics = []

    for ev in session:
        d = ev["data"]
        etype = ev["event"]

        if etype == "AGENT_START":
            metrics["input"] = d.get("input", "")

        elif etype == "LLM_METRIC":
            llm_metrics.append(d)
            metrics["total_tokens"]   += d.get("total_tokens", 0)
            metrics["total_latency_ms"] += d.get("latency_ms", 0)
            metrics["cost"]           += d.get("cost_estimate", 0)
            metrics["step_latencies"].append(d.get("latency_ms", 0))

        elif etype == "AGENT_TOOL_CALL":
            tool = d.get("tool", "")
            metrics["tool_calls"].append(tool)
            if tool not in VALID_TOOLS:
                metrics["hallucinated_tools"] += 1

        elif etype == "AGENT_FINAL_ANSWER":
            metrics["steps"]        = d.get("step", 0)
            metrics["final_answer"] = True

        elif etype == "AGENT_END":
            if not metrics["final_answer"]:
                metrics["timeout"] = True

    return metrics


def percentile(data: list, p: int) -> float:
    if not data: return 0
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s)-1)]


def print_report(sessions: dict):
    """In toàn bộ report theo EVALUATION.md."""

    # ── 1. Token Efficiency ───────────────────────────────────────
    header("📊 1. TOKEN EFFICIENCY")

    for mode in ["chatbot", "agent_v1", "agent_v2"]:
        slist = sessions[mode]
        if not slist:
            continue
        if mode == "chatbot":
            analyzed = [analyze_chatbot_session(s) for s in slist]
        else:
            analyzed = [analyze_agent_session(s) for s in slist]

        tokens = [a["total_tokens"] for a in analyzed if a["total_tokens"]]
        costs  = [a["cost"] for a in analyzed if a["cost"]]
        n = len(analyzed)

        label = {"chatbot":"💬 Chatbot", "agent_v1":"🤖 Agent v1", "agent_v2":"🛡️  Agent v2"}[mode]
        print(f"\n  {c(label, BOLD)}  ({n} sessions)")
        print(f"    Avg tokens / task  : {int(sum(tokens)/len(tokens)) if tokens else 'N/A'}")
        print(f"    Total tokens       : {sum(tokens)}")
        print(f"    Avg cost / task    : ${sum(costs)/len(costs):.4f}" if costs else "    Avg cost / task    : N/A")
        print(f"    Total cost         : ${sum(costs):.4f}" if costs else "    Total cost         : N/A")

    # ── 2. Latency ────────────────────────────────────────────────
    header("⏱  2. LATENCY ANALYSIS")

    for mode in ["chatbot", "agent_v1", "agent_v2"]:
        slist = sessions[mode]
        if not slist:
            continue
        if mode == "chatbot":
            analyzed = [analyze_chatbot_session(s) for s in slist]
            latencies = [a["latency_ms"] for a in analyzed if a["latency_ms"]]
        else:
            analyzed = [analyze_agent_session(s) for s in slist]
            latencies = [a["total_latency_ms"] for a in analyzed if a["total_latency_ms"]]

        if not latencies:
            continue

        label = {"chatbot":"💬 Chatbot", "agent_v1":"🤖 Agent v1", "agent_v2":"🛡️  Agent v2"}[mode]
        p50 = percentile(latencies, 50)
        p99 = percentile(latencies, 99)
        print(f"\n  {c(label, BOLD)}")
        print(f"    P50 latency        : {p50:,.0f} ms")
        print(f"    P99 latency        : {p99:,.0f} ms")
        print(f"    Min                : {min(latencies):,.0f} ms")
        print(f"    Max                : {max(latencies):,.0f} ms")
        print(f"    Avg                : {sum(latencies)/len(latencies):,.0f} ms")

    # ── 3. Loop Count ─────────────────────────────────────────────
    header("🔄 3. LOOP COUNT & TERMINATION QUALITY")

    for mode in ["agent_v1", "agent_v2"]:
        slist = sessions[mode]
        if not slist:
            continue
        analyzed = [analyze_agent_session(s) for s in slist]
        steps   = [a["steps"] for a in analyzed if a["steps"] > 0]
        finals  = sum(1 for a in analyzed if a["final_answer"])
        timeouts = sum(1 for a in analyzed if a["timeout"])
        n = len(analyzed)

        label = {"agent_v1":"🤖 Agent v1", "agent_v2":"🛡️  Agent v2"}[mode]
        print(f"\n  {c(label, BOLD)}  ({n} sessions)")
        print(f"    Avg steps / task   : {sum(steps)/len(steps):.1f}" if steps else "    N/A")
        print(f"    Min / Max steps    : {min(steps)} / {max(steps)}" if steps else "    N/A")
        print(f"    Clean terminations : {finals}/{n}  ({100*finals//n if n else 0}%)")
        print(f"    Timeouts           : {c(str(timeouts), RED if timeouts else GREEN)}/{n}")

    # ── 4. Failure Analysis ───────────────────────────────────────
    header("🚨 4. FAILURE ANALYSIS (Error Codes)")

    for mode in ["agent_v1", "agent_v2"]:
        slist = sessions[mode]
        if not slist:
            continue
        analyzed = [analyze_agent_session(s) for s in slist]

        timeouts    = sum(1 for a in analyzed if a["timeout"])
        halluc      = sum(a["hallucinated_tools"] for a in analyzed)
        parser_errs = sum(a["parser_errors"] for a in analyzed)
        n = len(analyzed)

        label = {"agent_v1":"🤖 Agent v1", "agent_v2":"🛡️  Agent v2"}[mode]
        print(f"\n  {c(label, BOLD)}  ({n} sessions)")
        print(f"    Timeout (max_steps): {c(str(timeouts), RED if timeouts else GREEN)}")
        print(f"    Hallucinated tools : {c(str(halluc), RED if halluc else GREEN)}")
        print(f"    Parser errors      : {c(str(parser_errs), RED if parser_errs else GREEN)}")

    # Guardrail blocks
    all_events = []
    for slist in sessions.values():
        for s in slist:
            all_events.extend(s)

    injections = sum(1 for e in all_events if e["event"] == "GUARDRAIL_INJECTION_BLOCKED")
    print(f"\n  {c('🛡️  Guardrail (Agent v2 only)', BOLD)}")
    print(f"    Injection attempts blocked : {c(str(injections), CYAN)}")

    # ── 5. Aggregate Reliability ──────────────────────────────────
    header("🏆 5. AGGREGATE RELIABILITY — v1 vs v2")

    print(f"\n  {'Metric':<35} {'Agent v1':>12} {'Agent v2':>12}")
    print("  " + "─" * 60)

    def metric_row(label, v1_val, v2_val, better="lower"):
        v1s = str(v1_val) if v1_val is not None else "N/A"
        v2s = str(v2_val) if v2_val is not None else "N/A"
        if v1_val is not None and v2_val is not None:
            if better == "lower":
                winner = c(v2s, GREEN) if v2_val <= v1_val else c(v2s, YELLOW)
                v1s    = c(v1s, YELLOW) if v2_val < v1_val else v1s
            else:
                winner = c(v2s, GREEN) if v2_val >= v1_val else c(v2s, YELLOW)
                v1s    = c(v1s, YELLOW) if v2_val > v1_val else v1s
        else:
            winner = v2s
        print(f"  {label:<35} {v1s:>12} {winner:>12}")

    def get_metric(mode, fn):
        slist = sessions[mode]
        if not slist: return None
        return fn([analyze_agent_session(s) for s in slist])

    v1_lat  = get_metric("agent_v1", lambda A: int(sum(a["total_latency_ms"] for a in A)/len(A)) if A else None)
    v2_lat  = get_metric("agent_v2", lambda A: int(sum(a["total_latency_ms"] for a in A)/len(A)) if A else None)
    v1_tok  = get_metric("agent_v1", lambda A: int(sum(a["total_tokens"] for a in A)/len(A)) if A else None)
    v2_tok  = get_metric("agent_v2", lambda A: int(sum(a["total_tokens"] for a in A)/len(A)) if A else None)
    v1_step = get_metric("agent_v1", lambda A: round(sum(a["steps"] for a in A)/len([a for a in A if a["steps"]]),1) if A else None)
    v2_step = get_metric("agent_v2", lambda A: round(sum(a["steps"] for a in A)/len([a for a in A if a["steps"]]),1) if A else None)
    v1_suc  = get_metric("agent_v1", lambda A: f"{sum(1 for a in A if a['final_answer'])}/{len(A)}")
    v2_suc  = get_metric("agent_v2", lambda A: f"{sum(1 for a in A if a['final_answer'])}/{len(A)}")
    v1_hall = get_metric("agent_v1", lambda A: sum(a["hallucinated_tools"] for a in A))
    v2_hall = get_metric("agent_v2", lambda A: sum(a["hallucinated_tools"] for a in A))
    v1_cost = get_metric("agent_v1", lambda A: f"${sum(a['cost'] for a in A):.4f}")
    v2_cost = get_metric("agent_v2", lambda A: f"${sum(a['cost'] for a in A):.4f}")

    metric_row("Avg latency / task (ms)",      v1_lat,  v2_lat,  better="lower")
    metric_row("Avg tokens / task",            v1_tok,  v2_tok,  better="lower")
    metric_row("Avg steps / task",             v1_step, v2_step, better="lower")
    metric_row("Successful terminations",      v1_suc,  v2_suc,  better="higher")
    metric_row("Hallucinated tool calls",      v1_hall, v2_hall, better="lower")
    metric_row("Total cost",                   v1_cost, v2_cost, better="lower")

    # Sessions count summary
    print(f"\n  Sessions analyzed: Chatbot={len(sessions['chatbot'])}, "
          f"Agent v1={len(sessions['agent_v1'])}, Agent v2={len(sessions['agent_v2'])}")

    # Compare chatbot vs agent
    sub("💬 Chatbot vs 🤖 Agent v1 — Head to Head")
    cb_slist = sessions["chatbot"]
    ag_slist = sessions["agent_v1"]
    if cb_slist and ag_slist:
        cb_analyzed = [analyze_chatbot_session(s) for s in cb_slist]
        ag_analyzed = [analyze_agent_session(s) for s in ag_slist]
        cb_tok = int(sum(a["total_tokens"] for a in cb_analyzed) / len(cb_analyzed))
        ag_tok = int(sum(a["total_tokens"] for a in ag_analyzed) / len(ag_analyzed))
        print(f"  Token overhead of Agent over Chatbot: {ag_tok}/{cb_tok} = {ag_tok/cb_tok:.1f}x")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="Path to specific log file")
    args = parser.parse_args()

    # Find log files
    if args.log:
        log_files = [args.log]
    else:
        log_files = sorted(glob.glob("logs/*.log"))
        if not log_files:
            print("❌ No log files found in logs/")
            sys.exit(1)

    print(c(f"\n📂 Log Analysis — {len(log_files)} file(s)", BOLD))
    for lf in log_files:
        print(f"  {lf}")

    all_events = []
    for lf in log_files:
        all_events.extend(load_logs(lf))

    print(f"\n  Total events: {len(all_events)}")

    sessions = group_sessions(all_events)
    print_report(sessions)


if __name__ == "__main__":
    main()
