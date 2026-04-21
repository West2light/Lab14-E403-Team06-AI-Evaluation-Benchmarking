# Báo cáo Cá nhân - Lab Day 14: AI Evaluation Benchmarking
**Họ tên:** Nguyễn Lê Trung  
**MSSV:** 2A202600174  
**Vai trò:** Nhóm Data — Vector Store Infrastructure & Synthetic Data Generation

---

## 1. Engineering Contribution (15đ)

### Module: Vector Store (`data/vector_store.py`)

Tôi xây dựng toàn bộ pipeline ingestion và retrieval từ đầu, thay thế placeholder không có code nào trong codebase ban đầu.

**Chunking pipeline:**

Tôi thiết kế hàm `_chunk_text()` với tham số `CHUNK_SIZE=400` ký tự và `CHUNK_OVERLAP=80` ký tự. Overlap đảm bảo thông tin nằm ở ranh giới hai chunk không bị cắt mất, quan trọng với tài liệu có cấu trúc bảng biểu (ví dụ SLA, Access Level).

**Canonical chunk ID với slugify:**

Vấn đề trọng tâm của toàn bộ pipeline là đảm bảo `ground_truth_doc_ids` trong golden set khớp chính xác với ID ChromaDB lưu. Source path trong tài liệu (`it/access-control-sop.md`) chứa ký tự `/` — không hợp lệ với ChromaDB. Tôi thiết kế hàm `make_chunk_id(source, idx)` để chuẩn hóa ID:

```python
def _slugify(source: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", source)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug

def make_chunk_id(source: str, idx: int) -> str:
    return f"{_slugify(source)}__chunk_{idx:03d}"
```

Hàm này được export và import bởi `synthetic_gen.py`, đảm bảo single source of truth cho ID — không thể xảy ra mismatch giữa ChromaDB và golden set.

**Ví dụ:**
- Source: `it/access-control-sop.md` → ID: `it-access-control-sop.md__chunk_000`
- Source: `hr/leave-policy-2026.pdf` → ID: `hr-leave-policy-2026.pdf__chunk_000`

**Kết quả:** 36 chunks từ 5 tài liệu, xác minh ChromaDB và code logic hoàn toàn đồng nhất.

**Expose `retrieve(query, top_k)`:**

```python
def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    ...
```

Hàm này được dùng bởi cả `engine/retrieval_eval.py` lẫn `agent/main_agent.py`.

---

### Module: Synthetic Data Generator (`data/synthetic_gen.py`)

Tôi implement toàn bộ pipeline sinh dữ liệu, thay thế placeholder trả về 1 cặp QA hardcoded.

**5 loại test case song song với HARD_CASES_GUIDE:**

| Generator | Loại | Số lượng |
|-----------|------|----------|
| `_gen_factual()` | `factual` | ~20 |
| `_gen_adversarial()` | `adversarial` | ~10 |
| `_gen_edge_out_of_context()` | `edge_out_of_context` | ~7 |
| `_gen_edge_ambiguous()` | `edge_ambiguous` | ~5 |
| `_gen_multiturn()` | `multi_turn` | ~8 |

Tất cả 5 generator chạy song song qua `asyncio.gather()`.

**Prompt engineering cho từng loại:**

- **Factual:** Yêu cầu LLM trích câu trả lời trực tiếp từ tài liệu, không được thêm thông tin ngoài.
- **Adversarial:** Phân 3 sub-type rõ ràng trong prompt — `prompt_injection`, `goal_hijacking`, `false_premise`. Expected answer mô tả hành vi đúng của agent (từ chối, sửa giả định sai), không phải câu trả lời nội dung.
- **Edge Out-of-Context:** Câu hỏi nghe có vẻ liên quan nhưng tài liệu không có thông tin — expected answer là mẫu "Tôi không có thông tin về vấn đề này".
- **Edge Ambiguous:** Câu hỏi thiếu ngữ cảnh chủ từ, thời gian — expected answer mô tả agent cần hỏi lại điều gì.
- **Multi-turn:** Thiết kế 2 lượt, turn 2 dùng đại từ tham chiếu ("nó", "điều đó") phụ thuộc turn 1. Lưu `turn_1` và `turn_1_answer` vào metadata để eval engine có thể tái tạo context.

**Schema mỗi record:**

```json
{
  "id": "tc_001",
  "question": "...",
  "expected_answer": "...",
  "context": "...",
  "ground_truth_doc_ids": ["it-access-control-sop.md__chunk_003"],
  "metadata": {
    "difficulty": "easy|medium|hard",
    "type": "factual|adversarial|edge_out_of_context|edge_ambiguous|multi_turn"
  }
}
```

---

### Module: Retrieval Evaluator (`engine/retrieval_eval.py`)

Tôi kết nối `evaluate_batch()` với `retrieve()` từ `vector_store.py`, thay thế mock data hardcoded:

```python
hits = retrieve(question, top_k=self.top_k)
retrieved_ids = [h["id"] for h in hits]
hr = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=self.top_k)
mrr = self.calculate_mrr(expected_ids, retrieved_ids)
```

Cases không có `ground_truth_doc_ids` (adversarial, OOC) được bỏ qua khỏi tính toán retrieval metrics, tránh làm sai lệch Hit Rate và MRR.

---

## 2. Technical Depth (15đ)

### Hit Rate và MRR

**Hit Rate** kiểm tra nhị phân: ít nhất 1 chunk đúng có xuất hiện trong top-k kết quả không. Đây là metric dễ giải thích nhưng không phân biệt được "chunk đúng ở vị trí 1" vs "chunk đúng ở vị trí k".

**MRR (Mean Reciprocal Rank)** giải quyết hạn chế đó:

```
MRR = (1/N) * Σ (1 / rank_i)
```

- Chunk đúng ở vị trí 1: đóng góp 1.0
- Chunk đúng ở vị trí 3: đóng góp 0.33
- Không tìm thấy: đóng góp 0

Trong RAG, MRR quan trọng hơn Hit Rate vì LLM đọc context theo thứ tự — chunk sai đứng đầu dù chunk đúng có trong top-5 vẫn có thể dẫn đến hallucination.

Mối liên hệ Retrieval ↔ Answer Quality: nếu `avg_hit_rate` cao nhưng answer score thấp, lỗi nằm ở generation (prompt, model). Nếu `avg_mrr` thấp dù `hit_rate` cao, lỗi nằm ở ranking — chunk đúng bị đẩy xuống sâu, LLM không đọc đến.

### Chunking Strategy và ảnh hưởng đến Retrieval

Tôi chọn `CHUNK_SIZE=400` dựa trên cấu trúc tài liệu: mỗi section trong docs có độ dài trung bình 300–500 ký tự. Chunk quá nhỏ (< 200) làm mất context liên câu; chunk quá lớn (> 800) làm embedding vector mang quá nhiều noise.

Overlap 80 ký tự (~20%) xử lý trường hợp câu hỏi cần thông tin trải dài qua ranh giới 2 chunk liền kề.

### Cohen's Kappa

Trong hệ thống multi-judge, `agreement_rate` nhị phân không loại bỏ xác suất đồng ý ngẫu nhiên. Cohen's Kappa bổ sung điều đó:

```
κ = (P_o - P_e) / (1 - P_e)
```

Với 2 judge đánh thang 1–5, nếu cả hai randomly chọn điểm 3, xác suất đồng ý là 1/5 = 0.2 — Kappa trừ bỏ phần đó. Kappa < 0.4 cho thấy 2 judge đang hiểu rubric khác nhau, cần calibration lại prompt judge.

### Position Bias

Position bias xuất hiện ở 2 tầng trong pipeline:

1. **Retrieval tầng:** Chunk sai đứng đầu trong kết quả → LLM sinh câu trả lời dựa vào chunk sai dù chunk đúng có trong top-5. Đây là lý do MRR quan trọng hơn Hit Rate.

2. **Judge tầng:** Nếu judge phải đọc 2 câu trả lời để so sánh, câu đứng trước thường được chấm điểm cao hơn. Phát hiện bằng cách swap thứ tự input, tính `score_delta` giữa 2 lần.

### Trade-off Chi phí vs. Chất lượng

Pipeline SDG gọi OpenAI API cho tất cả 50+ cases. Chi phí có thể giảm mà không giảm chất lượng bằng 2 cách:

- **Routing theo độ khó:** `factual/easy` case dùng `gpt-4o-mini`; `adversarial/hard` case mới dùng model mạnh hơn.
- **Cache embedding:** ChromaDB lưu embedding, chỉ embed lại khi tài liệu thay đổi — không gọi API embedding cho mỗi query.

Với ingestion hiện tại (36 chunks × `text-embedding-3-small`), chi phí embedding một lần ≈ $0.00002, không đáng kể. Chi phí chính nằm ở SDG (~50 API calls) và Judge (~100 API calls/benchmark run).

---

## 3. Problem Solving (10đ)

### Vấn đề 1: ChromaDB reject tên collection `kb` (< 3 ký tự)

**Phát hiện:** Chạy `python data/vector_store.py` lần đầu nhận `InvalidArgumentError: name: Expected a name containing 3-512 characters`.  
**Nguyên nhân:** `.env.example` đặt `CHROMA_COLLECTION=kb` — 2 ký tự, vi phạm constraint của ChromaDB.  
**Giải quyết:** Thêm guard trong code thay vì đổi config:

```python
_raw_collection = os.getenv("CHROMA_COLLECTION", "kb")
CHROMA_COLLECTION = _raw_collection if len(_raw_collection) >= 3 else f"{_raw_collection}_col"
```

Cách này backward-compatible: nếu team dùng `.env` với tên hợp lệ thì không bị ảnh hưởng.

### Vấn đề 2: ChromaDB có 72 chunks thay vì 36 sau khi re-ingest

**Phát hiện:** Sau khi fix slugify, chạy lại `ingest_documents(force=True)` — ChromaDB báo 72 chunks.  
**Nguyên nhân:** Lần ingest đầu dùng ID cũ chứa `/` (ví dụ `it/access-control-sop.md__chunk_000`), lần sau dùng ID slugified (`it-access-control-sop.md__chunk_000`). `upsert` không xóa ID cũ → 2 set tồn tại song song.  
**Giải quyết:** Xóa collection cũ và ingest lại từ đầu. Xác minh bằng cách list toàn bộ IDs từ ChromaDB và đối chiếu với output của code logic — kết quả khớp chính xác 36 chunks.

### Vấn đề 3: `ground_truth_doc_ids` trong golden set không khớp với ChromaDB IDs

**Phát hiện:** Khi trace flow cuối trước khi chạy benchmark, nhận ra `synthetic_gen.py` tự build ID theo cách riêng (dùng f-string với source gốc), trong khi `vector_store.py` đã có `make_chunk_id()` với slugify.  
**Nguyên nhân:** Hai file dùng 2 cách khác nhau để sinh ID — sẽ dẫn đến Hit Rate = 0 cho mọi case dù retrieval hoạt động đúng.  
**Giải quyết:** Export `make_chunk_id` từ `vector_store.py`, import vào `synthetic_gen.py`, thay thế toàn bộ chỗ build ID thủ công. Đây là nguyên tắc single source of truth cho ID — bất kỳ thay đổi nào về format ID chỉ cần sửa 1 nơi.
