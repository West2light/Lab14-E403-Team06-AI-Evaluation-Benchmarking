# Báo cáo cá nhân - AI Agent & LLM Judge

**Vai trò:** AI/Backend - Multi-Judge Consensus Engine

---

## 1. Engineering Contribution (15 điểm)

### 1.1. Xây dựng Multi-Judge Consensus Engine

Tôi phụ trách triển khai phần `engine/llm_judge.py`, thay thế judge mock ban đầu bằng hệ thống đánh giá có 2 model judge:

- `OpenAIJudge`: dùng OpenAI API với model mặc định `gpt-4o-mini`.
- `GeminiJudge`: dùng Gemini API với model mặc định `gemini-2.5-flash`.
- `HeuristicJudge`: fallback local khi API lỗi, thiếu key, sai key hoặc mất kết nối.
- `LLMJudge`: engine tổng hợp kết quả từ nhiều judge.

Hai judge dùng chung một rubric 1-5:

- `5`: đúng, đầy đủ, bám sát context.
- `4`: gần đúng, chỉ thiếu chi tiết nhỏ.
- `3`: đúng một phần nhưng thiếu hoặc bám context yếu.
- `2`: sai nhiều hoặc support kém.
- `1`: hallucination, không liên quan hoặc trái context.

Kết quả mỗi judge được chuẩn hóa về cùng schema:

```json
{
  "score": 1-5,
  "pass": true,
  "reason": "short explanation"
}
```

Hai judge được gọi song song bằng `asyncio.gather()`, sau đó engine tính:

- `final_score`
- `agreement_rate`
- `has_conflict`
- `score_gap`
- `conflict_strategy`

Nếu hai judge bất đồng mạnh, hệ thống dùng chiến lược bảo thủ:

```python
final_score = min(scores)
```

Nếu không có conflict, hệ thống lấy trung bình:

```python
final_score = sum(scores) / len(scores)
```

### 1.2. Tích hợp Agent V1 và V2 cho Regression Testing

Tôi triển khai hai phiên bản agent trong `agent/main_agent.py`:

- `AgentV1`: cùng flow với V2 nhưng cố tình lỗi retrieval.
- `AgentV2`: bản tối ưu dùng ChromaDB vector retrieval.

V1 không phải agent yếu hoàn toàn, mà mô phỏng lỗi thực tế trong hệ thống RAG: retrieval chọn sai context. Cụ thể, sau khi ChromaDB trả về danh sách chunks, V1 cố tình chọn các chunk có điểm thấp hơn:

```python
wrong_chunks = list(reversed(candidates))[: self.top_k]
```

V2 thì lấy top-k chunks tốt nhất từ ChromaDB:

```python
return [item for item in scored_chunks[: self.top_k] if item.score > 0]
```

Cách thiết kế này giúp regression test phản ánh đúng lỗi hệ thống: cùng pipeline generation, nhưng chất lượng khác nhau do retrieval.

---

## 2. Technical Depth (15 điểm)

### 2.1. MRR và Hit Rate

Hit Rate kiểm tra xem trong top-k retrieved chunks có ít nhất một chunk đúng hay không. Nếu có, Hit Rate = 1; nếu không, Hit Rate = 0.

MRR đo vị trí của chunk đúng đầu tiên trong danh sách retrieval:

```text
MRR = 1 / rank
```

Ví dụ:

- Chunk đúng ở vị trí 1: MRR = 1.0
- Chunk đúng ở vị trí 2: MRR = 0.5
- Không tìm thấy chunk đúng: MRR = 0

Trong hệ thống RAG, MRR quan trọng vì LLM thường ưu tiên context xuất hiện đầu tiên. Nếu context đúng bị đẩy xuống sâu hoặc không xuất hiện, agent dễ hallucinate hoặc trả lời sai.

### 2.2. Multi-Judge Agreement

Hệ thống dùng `agreement_rate` để đo mức đồng thuận giữa OpenAI judge và Gemini judge.

Logic hiện tại:

- Nếu hai judge không conflict và chênh lệch điểm nhỏ, `agreement_rate = 1.0`.
- Nếu hai judge bất đồng pass/fail hoặc lệch điểm lớn, `agreement_rate = 0.0`.

Conflict được xác định bằng:

```python
has_conflict = (len(set(passes)) > 1) or score_gap >= 2.0
```

Với bài lab, đây là một dạng calibration đơn giản. Nếu mở rộng, có thể thay bằng Cohen's Kappa để đo agreement tốt hơn vì Kappa loại bỏ xác suất đồng thuận ngẫu nhiên.

### 2.3. Cohen's Kappa

Cohen's Kappa đo mức đồng thuận giữa hai annotator hoặc hai judge, có tính đến khả năng đồng ý do ngẫu nhiên:

```text
kappa = (P_o - P_e) / (1 - P_e)
```

Trong bối cảnh multi-judge:

- `P_o`: tỷ lệ OpenAI và Gemini thật sự đồng ý.
- `P_e`: tỷ lệ đồng ý kỳ vọng do ngẫu nhiên.

Nếu Kappa thấp, nghĩa là hai judge có thể đang hiểu rubric khác nhau hoặc prompt judge chưa đủ rõ. Khi đó cần calibration lại prompt hoặc thêm judge thứ ba.

### 2.4. Position Bias

Position bias là hiện tượng LLM bị ảnh hưởng bởi vị trí thông tin trong prompt.

Trong project này, position bias có thể xuất hiện ở hai nơi:

1. Retrieval context:
   - Nếu chunk sai đứng đầu, agent có thể dựa vào chunk sai để trả lời.
   - Đây chính là lỗi được mô phỏng ở `AgentV1`.

2. Judge evaluation:
   - Nếu judge phải so sánh nhiều câu trả lời, câu trả lời đứng trước có thể được ưu ái hơn.
   - Hướng mở rộng là dùng `check_position_bias()` để đảo thứ tự response và đo `score_delta`.

### 2.5. Trade-off giữa chi phí và chất lượng

Hệ thống hiện dùng 2 LLM judge nên chất lượng đánh giá khách quan hơn một judge đơn lẻ, nhưng chi phí cũng cao hơn.

Trade-off chính:

- Dùng 2 judge giúp giảm bias và phát hiện conflict.
- Dùng model lớn cho mọi case tăng chi phí.
- Có thể giảm chi phí bằng fallback hoặc routing:
  - Case dễ: dùng model nhỏ hoặc heuristic trước.
  - Case conflict: mới gọi model mạnh hơn hoặc judge thứ ba.
  - Case có score gap thấp: lấy trung bình luôn, không cần thêm đánh giá.

Trong implementation hiện tại, fallback local giúp pipeline không crash khi API lỗi, đồng thời vẫn duy trì output đúng schema.

---

## 3. Problem Solving (10 điểm)

### Vấn đề 1: Tích hợp ChromaDB Retrieval

Ban đầu agent retrieve trực tiếp từ `data/golden_set.jsonl` bằng lexical overlap. Tôi đã sửa để agent ưu tiên dùng vector retrieval từ ChromaDB:

```python
from vector_store import ingest_documents, retrieve as chroma_retrieve
```

Trong `MainAgent`, ChromaDB được khởi tạo một lần ở constructor:

```python
self._chroma_retrieve = self._init_chroma_retriever()
```

Điều này giải quyết lỗi trước đó: `ingest_documents(force=False)` bị gọi lặp lại ở mỗi query, làm terminal in nhiều dòng:

```text
[VectorStore] Collection 'kb_col' already has ... chunks. Skipping ingest.
```

Sau khi sửa, agent chỉ kiểm tra ChromaDB một lần khi khởi tạo.

### Vấn đề 2: API có thể lỗi, sai key hoặc mất mạng

Trong quá trình chạy, OpenAI hoặc Gemini có thể lỗi:

- Sai API key.
- Connection error.
- Gemini quá tải.
- Response không phải JSON hợp lệ.

Giải pháp:

- Bọc API call bằng `try/except`.
- Parse JSON an toàn bằng `_extract_json_object()`.
- Nếu API lỗi thì dùng `HeuristicJudge`.
- Ghi rõ `used_fallback` và reason trong kết quả judge.

Điều này giúp benchmark không bị dừng giữa chừng.

---
