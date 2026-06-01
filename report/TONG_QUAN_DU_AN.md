# Báo Cáo Tổng Quan: Dự Án ReAct Agent E-Commerce

> **Ngày**: 2026-06-01 | **Model**: GPT-4o | **Lab**: Day 3 — Chatbot vs ReAct Agent

---

## 1. Bức Tranh Lớn — Tại Sao Cần Agent?

### Vấn đề với LLM Chatbot thông thường

Hãy tưởng tượng bạn hỏi một nhân viên tư vấn bán hàng:

> *"Tôi muốn mua 2 iPhone 15, dùng coupon WINNER, ship về Hà Nội. Tổng bao nhiêu tiền?"*

Một nhân viên **không có quyền truy cập hệ thống** sẽ trả lời kiểu:

> *"Ước tính khoảng $1,578 USD, tuy nhiên giá có thể thay đổi tùy retailer..."*

Trong khi đó, một nhân viên **có máy tính + hệ thống POS** sẽ:
1. Tra kho → iPhone 15: còn 10 chiếc ✅
2. Tra giá → 25,000,000 VND/chiếc
3. Kiểm tra coupon WINNER → giảm 10%
4. Tính phí ship Hà Nội → 30,000 VND
5. Tính tổng → **45,030,000 VND**

**LLM Chatbot = nhân viên không có hệ thống.**
**ReAct Agent = nhân viên CÓ hệ thống + biết cách dùng nó.**

---

## 2. ReAct Agent Là Gì?

**ReAct** = **Re**asoning + **Act**ing

Đây là một kiến trúc (pattern) ra đời năm 2022 từ paper của Google DeepMind, kết hợp hai khả năng:

| Khả năng | Ý nghĩa |
|---|---|
| **Reasoning** (Suy luận) | LLM tư duy từng bước, lập kế hoạch trước khi hành động |
| **Acting** (Hành động) | Gọi các công cụ (tools) để lấy thông tin thực tế từ thế giới bên ngoài |

### Chu trình ReAct

```
┌─────────────────────────────────────────────────────────┐
│                    ReAct Loop                           │
│                                                         │
│   User Input                                            │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────────────────────────────┐                   │
│   │  Thought: Tôi cần kiểm tra giá  │  ← LLM suy nghĩ  │
│   └─────────────────────────────────┘                   │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────────────────────────────┐                   │
│   │  Action: get_product_price(...)  │  ← LLM quyết định│
│   └─────────────────────────────────┘                   │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────────────────────────────┐                   │
│   │  Observation: 25,000,000 VND    │  ← Tool trả về   │
│   └─────────────────────────────────┘                   │
│       │                                                 │
│       ▼  (lặp lại nếu cần thêm thông tin)               │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────────────────────────────┐                   │
│   │  Final Answer: Tổng là 45M VND  │  ← Kết quả cuối  │
│   └─────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

Điểm mấu chốt: **LLM không biết gì trước — nó phải tự hỏi thông tin từng bước.** Giống như khi bạn giải toán: bạn không nhìn vào đáp án ngay, bạn viết từng bước ra giấy nháp.

---

## 3. Kiến Trúc Hệ Thống

### Sơ đồ tổng thể

```
┌──────────────────────────────────────────────────────────────────┐
│                        HỆTHỐNG AGENT                            │
│                                                                  │
│  ┌─────────┐    ┌──────────────┐    ┌────────────────────────┐  │
│  │  main.py │───►│  LLMProvider │    │     ReActAgent         │  │
│  └─────────┘    │  (interface) │    │                        │  │
│                 └──────┬───────┘    │  run(user_input)       │  │
│                        │            │    ├── get_system_prompt│  │
│           ┌────────────┼─────────┐  │    ├── _parse_action   │  │
│           ▼            ▼         ▼  │    ├── _execute_tool   │  │
│    ┌────────────┐ ┌────────┐ ┌──────┐   │    └── _parse_kwargs   │  │
│    │  OpenAI    │ │ Gemini │ │Local │◄──┤                        │  │
│    │  Provider  │ │ Provider│ │(CPU) │  └────────────────────────┘  │
│    └────────────┘ └────────┘ └──────┘                              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    TOOLS REGISTRY                           │ │
│  │  check_stock │ get_price │ get_discount │ shipping │ total  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │  SimpleChatbot   │  │         Telemetry System             │ │
│  │  (baseline)      │  │  logger.py + metrics.py → logs/*.log │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Các thành phần chính

#### `LLMProvider` — Interface đa nhà cung cấp

```
LLMProvider (Abstract Base Class)
    ├── generate(prompt, system_prompt) → Dict {content, usage, latency_ms}
    └── stream(prompt, system_prompt)  → Generator[str]

Implementations:
    ├── OpenAIProvider   → gọi api.openai.com
    ├── GeminiProvider   → gọi Google AI Studio
    └── LocalProvider    → chạy model .gguf trực tiếp trên CPU
```

Thiết kế kiểu **Strategy Pattern** — Agent không cần biết đang dùng OpenAI hay Gemini, chỉ cần gọi `llm.generate()`.

#### `ReActAgent` — Bộ não điều phối

```python
class ReActAgent:
    llm: LLMProvider          # Ai trả lời?
    tools: List[Dict]         # Có những công cụ gì?
    max_steps: int = 6        # Giới hạn vòng lặp (circuit breaker)
    history: List             # Lịch sử hội thoại
```

#### `TOOLS_REGISTRY` — Hộp công cụ của Agent

Mỗi tool là một Python function thuần túy:

| Tool | Input | Output |
|---|---|---|
| `check_stock(item_name)` | Tên sản phẩm | Số lượng tồn kho |
| `get_product_price(item_name)` | Tên sản phẩm | Giá (VND) |
| `get_discount(coupon_code)` | Mã coupon | % giảm giá |
| `calc_shipping(weight_kg, destination)` | Cân nặng + thành phố | Phí vận chuyển (VND) |
| `calculate_total(price, qty, discount, shipping)` | 4 tham số số | Bảng tổng kết đơn hàng |

#### `Telemetry` — Hệ thống giám sát

Mỗi lần gọi LLM, hệ thống ghi log tự động:
```json
{
  "timestamp": "2026-06-01T07:27:53",
  "event": "LLM_METRIC",
  "data": {
    "provider": "openai",
    "model": "gpt-4o",
    "prompt_tokens": 600,
    "completion_tokens": 27,
    "total_tokens": 627,
    "latency_ms": 5166,
    "cost_estimate": 0.003405
  }
}
```

---

## 4. Luồng Hoạt Động Chi Tiết — Ví Dụ TC-03

**Câu hỏi**: *"Tôi muốn mua 2 iPhone 15 dùng coupon WINNER ship về Hà Nội. Tổng bao nhiêu?"*

### Bước 1 — LLM lên kế hoạch

```
[System Prompt]: Bạn là E-commerce Assistant, có các tools: check_stock, get_product_price...
[User]: Tôi muốn mua 2 iPhone 15...

[LLM Output]:
Thought: I need to get the price of iPhone 15 first.
Action: get_product_price(item_name=iPhone 15)
```

### Bước 2 — Agent thực thi tool, nối Observation vào prompt

```
→ _parse_action(): tool_name="get_product_price", raw_args="item_name=iPhone 15"
→ _parse_kwargs(): {"item_name": "iPhone 15"}
→ get_product_price("iPhone 15") = "Unit price: 25,000,000 VND"

[Conversation nối tiếp]:
Observation: Unit price of 'iPhone 15': 25,000,000 VND.
```

### Bước 3, 4, 5 — Lặp lại

```
Thought: I have the price. Now I need the discount for coupon WINNER.
Action: get_discount(coupon_code=WINNER)
Observation: Coupon 'WINNER' is valid. Discount: 10% off.

Thought: Now I need shipping cost to Hanoi.
Action: calc_shipping(weight_kg=0.5, destination=hanoi)
Observation: Shipping cost to hanoi for 0.5 kg: 30,000 VND.

Thought: I have all info. Let me calculate the total.
Action: calculate_total(unit_price=25000000, quantity=2, discount_pct=10, shipping_cost=30000)
Observation: 
  Subtotal (2 × 25,000,000): 50,000,000 VND
  Discount (10%): -5,000,000 VND
  Shipping: 30,000 VND
  TOTAL: 45,030,000 VND
```

### Bước cuối — Final Answer

```
Thought: I have the complete calculation. I can now answer.
Final Answer: The total cost for 2 iPhone 15s with coupon WINNER
              shipped to Hanoi is 45,030,000 VND.
```

**Tổng số bước**: 4 tool calls (4 LLM turns), ~5,096ms

---

## 5. So Sánh Thực Nghiệm — Số Liệu Thực Từ Run

### Kết quả chạy ngày 2026-06-01

| Test Case | Câu hỏi | Chatbot | Agent | Người Chiến Thắng |
|---|---|---|---|---|
| TC-01 Simple | iPhone 15 còn hàng không? | Trả lời chung chung | **10 đơn vị** (chính xác) | Agent ✅ |
| TC-02 Medium | MacBook Pro giá và tồn kho? | Đoán ~$1,299 USD | **45M VND, 3 units** | Agent ✅ |
| TC-03 Hard | Mua 2 iPhone + coupon + ship | Đoán $1,578 USD (sai) | **45,030,000 VND** (đúng) | Agent ✅ |
| TC-04 Error | Coupon FAKE99 có dùng được không? | "Thử apply xem sao" | Phát hiện invalid + tính tiếp | Agent ✅ |

### Số liệu Telemetry tổng phiên

| Chỉ số | Giá trị |
|---|---|
| Tổng lần gọi LLM | 18 calls |
| Tổng tokens tiêu thụ | 11,511 tokens |
| Tổng chi phí ước tính | $0.0728 USD |
| Latency trung bình/call | 1,760 ms |
| Tổng tool calls Agent thực hiện | 10 calls |
| Tỉ lệ thành công Final Answer | 4/4 (100%) |

### Token so sánh: Chatbot vs Agent (TC-03)

```
Chatbot (TC-03):
  Prompt:     92 tokens
  Completion: 360 tokens
  Total:      452 tokens  ← ít token hơn nhưng SAI

Agent (TC-03, 4 turns tổng):
  Turn 1: 618 + 30  = 648 tokens  (get_price)
  Turn 2: 668 + 32  = 700 tokens  (get_discount)
  Turn 3: ...
  Total:  ~2,700 tokens  ← nhiều token hơn nhưng ĐÚNG
```

**Insight**: Agent tốn token nhiều hơn 6x so với Chatbot trên multi-step task, nhưng chất lượng không thể so sánh. Đây là trade-off điển hình của agentic systems.

---

## 6. Cơ Chế Phòng Chống Vòng Lặp Vô Hạn

Đây là vấn đề nghiêm trọng trong thực tế: nếu LLM cứ gọi tool lặp đi lặp lại mà không tìm được Final Answer → **chi phí API tăng vô hạn**.

### 3 lớp bảo vệ trong code

**Lớp 1 — `max_steps` circuit breaker** (cứng nhất):
```python
for step in range(self.max_steps):    # ← Vòng lặp CÓ GIỚI HẠN
    ...
else:
    # Chỉ chạy nếu hết max_steps mà không có Final Answer
    return "Could not complete within max steps."
```

**Lớp 2 — Nudge khi không có Action**:
```python
if action is None:
    conversation += "Observation: No action detected. Please call a tool or give Final Answer.\n"
    continue    # ← Nhắc LLM tiếp tục thay vì bỏ qua
```

**Lớp 3 — System Prompt enforcement**:
```
"When you have enough information, write Final Answer: immediately.
Never loop more than necessary."
```

---

## 7. Điểm Khác Biệt Cốt Lõi

```
┌─────────────────────────────────────┬───────────────────────────────────────┐
│           CHATBOT                   │           ReACT AGENT                 │
├─────────────────────────────────────┼───────────────────────────────────────┤
│ Trả lời từ knowledge khi training   │ Truy vấn thông tin thực tế           │
│ 1 LLM call duy nhất                │ Nhiều LLM call (mỗi step 1 call)      │
│ Không có "trí nhớ" trong phiên     │ Conversation tích lũy dần             │
│ Không thể tính toán chính xác       │ Dùng tool Calculator để tính          │
│ Nhanh + rẻ cho câu hỏi đơn giản    │ Chậm + tốn token hơn                  │
│ Hallucinate khi thiếu thông tin     │ Biết mình không biết → hỏi tool       │
│ Không thể recover khi gặp lỗi      │ Adaptive: điều chỉnh khi tool trả lỗi │
│ Không có audit trail               │ Mọi bước đều được log đầy đủ          │
└─────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 8. Hướng Phát Triển Tiếp Theo

### Nâng cấp gần (v2)

- **Cải thiện System Prompt**: Thêm few-shot examples cho từng tool để giảm JSON parsing errors
- **Error recovery**: Nếu tool trả về lỗi, Agent tự thử lại với arguments khác nhau
- **Conversation memory**: Lưu lịch sử session để support multi-turn hội thoại

### Nâng cấp xa (Production)

- **Vector DB Tool Retrieval**: 50+ tools → chỉ chọn top-K tool relevant bằng semantic search
- **Async Tool Calls**: Các tool độc lập chạy song song với `asyncio.gather()`  
- **Multi-Agent**: Orchestrator Agent điều phối nhiều Sub-Agent chuyên biệt
- **LangGraph**: Thay loop đơn giản bằng state machine có branching, retry, và persistence

---

## 9. Kết Luận

Dự án này xây dựng một **Smart E-Commerce Assistant** sử dụng kiến trúc ReAct Agent, minh họa sự khác biệt căn bản giữa:

- **Chatbot**: Nói chuyện dựa trên ký ức → giỏi Q&A chung, kém multi-step
- **ReAct Agent**: Suy nghĩ + hành động + quan sát → giỏi tác vụ phức tạp cần dữ liệu thực

Con Agent trong lab này không chỉ là một chatbot "thông minh hơn" — nó là một **hệ thống lập luận tự động** có khả năng:
1. Tự phân tích yêu cầu phức tạp thành các bước nhỏ
2. Chọn và sử dụng đúng công cụ cho từng bước
3. Xử lý lỗi và thích nghi khi công cụ trả về kết quả ngoài mong đợi
4. Dừng đúng lúc khi đã có đủ thông tin

Đây chính là nền tảng của các hệ thống AI tự động (Agentic AI) đang được triển khai trong ngành công nghiệp ngày nay.
