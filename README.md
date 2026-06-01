# Lab 3: Chatbot vs ReAct Agent (Industry Edition)

Welcome to Phase 3 of the Agentic AI course! This lab focuses on moving from a simple LLM Chatbot to a sophisticated **ReAct Agent** with industry-standard monitoring.

## 🚀 Getting Started

### 1. Setup Environment
Copy the `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

### 2. Install Dependencies
```bash
# Cài đầy đủ (bao gồm local model support)
pip install -r requirements.txt

# Hoặc cài nhanh (chỉ cần OpenAI/Gemini + Web UI)
pip install openai google-generativeai python-dotenv flask pytest
```

### 3. Directory Structure
- `src/agent/`    — ReAct loop (`agent.py`) và guardrails (`guardrails.py`)
- `src/chatbot.py` — Baseline Chatbot (không có tool)
- `src/tools/`    — 9 tools chia 2 nhóm: ORDER + DISCOVERY
- `src/core/`     — LLM providers (OpenAI, Gemini, Local)
- `src/telemetry/` — Token tracking & structured logging
- `web/`          — Flask backend + HTML UI
- `tests/`        — Unit tests (31 tests, không cần API)
- `experiments/`  — Ablation study scripts

---

## ▶️ Cách Chạy

### Cách 1 — So sánh 3-way (Chatbot vs Agent v1 vs Agent v2)
```bash
python main.py              # chạy tất cả 3 mode trên 4 test cases
python main.py --chatbot    # chỉ Chatbot
python main.py --agent      # chỉ Agent v1 (ReAct)
python main.py --v2         # chỉ Agent v2 (Guardrails + Retry)
```

### Cách 2 — Terminal Chat (gõ câu hỏi trực tiếp)
```bash
python chat.py              # Agent v1 (mặc định)
python chat.py --v2         # Agent v2 với guardrails
python chat.py --chatbot    # Chatbot baseline (không có tool)
```
> Trong chat: gõ `/help` xem hướng dẫn, `/db` xem database, `/quit` để thoát.

### Cách 3 — 🌐 Web UI (Giao diện trình duyệt)
```bash
# Bước 1: Cài Flask (nếu chưa có)
pip install flask

# Bước 2: Khởi động server
python web/server.py

# Bước 3: Mở trình duyệt
# http://localhost:5000
```

**Web UI có:**
- 💬 Chat real-time với Agent
- 🧠 Panel **Agent Thinking** — xem từng bước `Thought → Action → Observation` trực tiếp
- 🔄 Chuyển đổi giữa **Agent v1 / Agent v2 / Chatbot** bằng 1 click
- 🗄️ Sidebar: database sản phẩm, coupon, shipping
- 💡 Quick prompts để thử nhanh không cần gõ

### Cách 4 — Unit Tests
```bash
# Chạy ngay, không cần API key (31 tests)
python -m pytest tests/test_agent_chatbot.py -v -m "not integration"

# Tất cả tests bao gồm integration (cần API key)
python -m pytest tests/test_agent_chatbot.py -v
```

### Cách 5 — Ablation Study
```bash
python experiments/ablation_study.py --dry-run   # xem kế hoạch, không tốn token
python experiments/ablation_study.py --exp A     # Prompt v1 vs v2
python experiments/ablation_study.py --exp B     # 5 tools vs 9 tools
python experiments/ablation_study.py --exp C     # Agent v1 vs v2 (guardrails)
python experiments/ablation_study.py             # chạy tất cả
```

---

## 🏠 Running with Local Models (CPU)

If you don't want to use OpenAI or Gemini, you can run Phi-3 directly on your CPU.

### 1. Download the Model
- [Phi-3-mini-4k-instruct-GGUF](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
- Direct: [phi-3-mini-4k-instruct-q4.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf) (~2.2GB)

### 2. Place Model
```
models/
└── Phi-3-mini-4k-instruct-q4.gguf
```

### 3. Update `.env`
```env
DEFAULT_PROVIDER=local
LOCAL_MODEL_PATH=./models/Phi-3-mini-4k-instruct-q4.gguf
```

---

## 🎯 Lab Objectives

1. **Baseline Chatbot** — Observe limitations of LLM without tools (hallucination on multi-step queries)
2. **ReAct Loop** — `Thought → Action → Observation` cycle in `src/agent/agent.py`
3. **Tool Design** — 9 tools across 2 domains: ORDER (5) + DISCOVERY (4)
4. **Agent v2** — Production hardening: InputGuardrail + RetryHandler + OutputGuardrail
5. **Evaluation** — Head-to-head comparison with telemetry, token cost, latency

## 🛠️ Architecture
```
User Input
    │
    ▼ InputGuardrail (Agent v2 only)
    │
    ▼ ReAct Loop (max 6 steps)
    │   Thought → Action: tool(args) → Observation → repeat
    │
    ▼ OutputGuardrail (Agent v2 only)
    │
    ▼ Final Answer
```

---

*Happy Coding! Let's build agents that actually work.*
