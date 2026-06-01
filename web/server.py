"""
web/server.py — Flask Backend for Agent Web UI

Chạy: python web/server.py
Mở:  http://localhost:5000
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, Response, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))


def build_agent(mode: str = "agent"):
    from src.core.openai_provider import OpenAIProvider
    from src.tools import TOOLS_REGISTRY
    llm = OpenAIProvider(
        model_name=os.getenv("DEFAULT_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    if mode == "chatbot":
        from src.chatbot import SimpleChatbot
        return SimpleChatbot(llm=llm), mode
    elif mode == "v2":
        from src.agent.guardrails import RobustReActAgent
        return RobustReActAgent(llm=llm, tools=TOOLS_REGISTRY, max_steps=6, max_retries=3), mode
    else:
        from src.agent.agent import ReActAgent
        return ReActAgent(llm=llm, tools=TOOLS_REGISTRY, max_steps=6), mode


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    body     = request.json or {}
    question = body.get("message", "").strip()
    mode     = body.get("mode", "agent")

    if not question:
        return jsonify({"error": "Empty message"}), 400

    def stream():
        try:
            agent, agent_mode = build_agent(mode)

            if agent_mode == "chatbot":
                yield sse({"type": "thinking_start"})
                t0  = time.time()
                ans = agent.chat(question)
                ms  = int((time.time() - t0) * 1000)
                yield sse({"type": "thought", "content": "Generating response (no tools)...", "step": 1, "ms": ms})
                yield sse({"type": "final", "content": ans, "steps": 1, "total_ms": ms})
                return

            from src.agent.agent import _parse_action, _parse_final_answer

            yield sse({"type": "thinking_start"})
            conversation   = f"User Question: {question}\n\n"
            system_prompt  = agent.get_system_prompt()
            total_start    = time.time()

            for step in range(agent.max_steps):
                step_n = step + 1
                t0     = time.time()
                result = agent.llm.generate(prompt=conversation, system_prompt=system_prompt)
                ms     = int((time.time() - t0) * 1000)
                llm_output = result.get("content", "")
                tokens     = result.get("usage", {}).get("total_tokens", "?")
                conversation += llm_output + "\n"

                for line in llm_output.strip().split("\n"):
                    line = line.strip()
                    if line.lower().startswith("thought:"):
                        yield sse({"type": "thought", "content": line[8:].strip(),
                                   "step": step_n, "ms": ms, "tokens": tokens})
                    elif line.lower().startswith("action:"):
                        yield sse({"type": "action", "content": line[7:].strip(), "step": step_n})
                    elif line.lower().startswith("final answer:"):
                        yield sse({"type": "final_line", "content": line[13:].strip()})

                final = _parse_final_answer(llm_output)
                if final:
                    total_ms = int((time.time() - total_start) * 1000)
                    yield sse({"type": "final", "content": final,
                               "steps": step_n, "total_ms": total_ms, "tokens": tokens})
                    return

                action = _parse_action(llm_output)
                if action:
                    tool_name, raw_args = action
                    obs = agent._execute_tool(tool_name, raw_args)
                    yield sse({"type": "observation", "tool": tool_name,
                               "args": raw_args, "result": obs, "step": step_n})
                    conversation += f"Observation: {obs}\n\n"
                else:
                    conversation += "Observation: No action. Please call a tool or write Final Answer.\n\n"

            yield sse({"type": "error", "content": "Agent reached max steps without a final answer."})

        except Exception as e:
            yield sse({"type": "error", "content": str(e)})

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/db")
def get_db():
    return jsonify({
        "products": [
            {"name": "iPhone 15",   "price": 25_000_000, "stock": 10,  "category": "smartphone"},
            {"name": "iPhone 14",   "price": 18_000_000, "stock": 5,   "category": "smartphone"},
            {"name": "Samsung S24", "price": 22_000_000, "stock": 0,   "category": "smartphone"},
            {"name": "MacBook Pro", "price": 45_000_000, "stock": 3,   "category": "laptop"},
            {"name": "AirPods Pro", "price":  7_000_000, "stock": 15,  "category": "audio"},
            {"name": "iPad",        "price": 20_000_000, "stock": 8,   "category": "tablet"},
            {"name": "Laptop Asus", "price": 15_000_000, "stock": 12,  "category": "laptop"},
        ],
        "coupons": [
            {"code": "WINNER",  "discount": 10, "note": "Áp dụng tất cả"},
            {"code": "SALE20",  "discount": 20, "note": "iPhone 14, Asus..."},
            {"code": "VIP50",   "discount": 50, "note": "iPhone 15, MacBook"},
            {"code": "NEWUSER", "discount": 15, "note": "Áp dụng tất cả"},
        ],
        "shipping": [
            {"city": "Hà Nội",  "key": "hanoi",  "fee": 30_000},
            {"city": "TP.HCM",  "key": "hcm",    "fee": 50_000},
            {"city": "Đà Nẵng", "key": "danang", "fee": 40_000},
            {"city": "Cần Thơ", "key": "cantho", "fee": 60_000},
            {"city": "Huế",     "key": "hue",    "fee": 45_000},
        ],
    })


@app.route("/api/tools")
def get_tools():
    from src.tools import TOOLS_REGISTRY
    return jsonify([{"name": t["name"], "description": t["description"]} for t in TOOLS_REGISTRY])


if __name__ == "__main__":
    print("\n🚀 Agent Web UI starting...")
    print("   Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
