# Báo cáo Cá nhân - Lab Day 14: AI Evaluation Benchmarking
**Họ tên:** Vương Hoàng Giang  
**MSSV:** 2A202600349  
**Vai trò:** Data Preparation & Quality Assurance

---

## 1. Engineering Contribution (15đ)

### Đóng góp cụ thể

**Module: Synthetic Data Generator (`data/synthetic_gen.py`)**

Tôi chịu trách nhiệm thiết kế và mở rộng pipeline sinh dữ liệu test tự động. Cụ thể:

- Phân tích `HARD_CASES_GUIDE.md` và xác định 4 loại test case còn thiếu so với yêu cầu
- Thêm 4 prompt template và 4 async generator function mới:
  - `EDGE_CONFLICTING_PROMPT` + `_gen_edge_conflicting()`: chọn 2 chunk từ nguồn khác nhau, yêu cầu LLM tạo câu hỏi lộ mâu thuẫn
  - `MULTITURN_CORRECTION_PROMPT` + `_gen_multiturn_correction()`: tình huống người dùng đính chính giữa cuộc hội thoại
  - `LATENCY_STRESS_PROMPT` + `_gen_latency_stress()`: câu hỏi tổng hợp toàn bộ tài liệu dài (~3000 ký tự context)
  - `COST_EFFICIENCY_PROMPT` + `_gen_cost_efficiency()`: câu hỏi đơn giản với `max_expected_tokens: 30` để đo token dư thừa
- Cập nhật `main()` để chạy song song cả 9 generator với `asyncio.gather()`
- Fix bug type-narrowing: `isinstance(res, Exception)` → `isinstance(res, BaseException)`

Kết quả: golden_set tăng từ 50 lên **66 test cases**, bao phủ đầy đủ 4 nhóm trong guide.

**Module: Vector Store (`data/vector_store.py`)**

- Fix `CHROMA_DB_PATH` từ đường dẫn tương đối `"./chroma_db"` sang tuyệt đối dựa theo `Path(__file__)`, đảm bảo hoạt động đúng bất kể thư mục chạy lệnh.

**Git commits liên quan:** https://github.com/VinUni-AI20k/Lab14-AI-Evaluation-Benchmarking/commit/7ce10a487d81747a57a20ad989c053ea80df9369
- `check data and update data` — kiểm tra và cập nhật golden_set
- `Update DataSet` — bổ sung test cases mới

---

## 2. Technical Depth (15đ)

### MRR (Mean Reciprocal Rank)
MRR đo chất lượng retrieval theo thứ hạng của chunk đúng đầu tiên trong danh sách kết quả trả về. Công thức: `MRR = (1/N) * Σ(1/rank_i)`. Ví dụ: nếu chunk đúng đứng ở vị trí 2, MRR contribution = 0.5. Chỉ số này phản ánh việc agent có đưa thông tin quan trọng nhất lên đầu không — ảnh hưởng trực tiếp đến chất lượng câu trả lời vì LLM thường ưu tiên context đầu tiên (position bias).

### Cohen's Kappa
Hệ số đo mức độ đồng thuận giữa 2 judge model, loại bỏ yếu tố ngẫu nhiên. Công thức: `κ = (P_o - P_e) / (1 - P_e)`. Trong hệ thống multi-judge, nếu κ < 0.6 nghĩa là 2 model đang đánh giá theo tiêu chí khác nhau đáng kể — cần xem lại prompt của judge hoặc thêm calibration step để đồng bộ tiêu chí.

### Position Bias
Hiện tượng LLM judge cho điểm cao hơn cho câu trả lời xuất hiện ở vị trí đầu tiên trong prompt, bất kể chất lượng thực tế. Trong hệ thống này, position bias ảnh hưởng theo 2 chiều: (1) retriever trả về chunk sai ở đầu → LLM sinh câu trả lời sai; (2) judge đọc câu trả lời của model A trước model B → judge thiên vị model A. Giải pháp: swap thứ tự input khi judge và lấy trung bình.

### Trade-off Chi phí vs. Chất lượng
Golden set có loại `cost_efficiency` với field `max_expected_tokens: 30`. Với câu hỏi đơn giản (hỏi một con số, một ngày), nếu agent trả lời 200 token thay vì 10 token thì lãng phí 20x chi phí mà không tăng chất lượng. Trong thực tế, dùng model nhỏ hơn (gpt-4o-mini thay vì gpt-4o) cho factual lookup có thể giảm 70-80% chi phí với chỉ ~5% giảm accuracy.

---

## 3. Problem Solving (10đ)

### Vấn đề 1: Golden set thiếu 4 loại hard case theo guide
**Phát hiện:** So sánh các `type` trong file JSONL với `HARD_CASES_GUIDE.md`, xác định thiếu `edge_conflicting`, `multi_turn_correction`, `latency_stress`, `cost_efficiency`.  
**Giải quyết:** Thiết kế prompt riêng cho từng loại, chú ý đặc thù từng case: `edge_conflicting` cần 2 nguồn chunk khác nhau; `cost_efficiency` cần thêm field `max_expected_tokens` để eval engine đo được.

### Vấn đề 2: File golden_set.jsonl bị ghi 0 bytes sau khi chạy script
**Phát hiện:** `wc -l data/golden_set.jsonl` trả về 0 dù script in "64 test cases saved".  
**Nguyên nhân:** Script chạy lần trước bị interrupt sau khi `open(path, "w")` truncate file nhưng trước khi ghi — output terminal là từ lần chạy thành công trước đó.  
**Giải quyết:** Chạy lại script, xác nhận bằng `ls -la` (file size > 0) thay vì chỉ dùng `wc -l`.

### Vấn đề 3: edge_conflicting cases không thực sự mâu thuẫn
**Phát hiện:** Sau khi đọc lại 3 case `edge_conflicting` (tc_051–tc_053), nhận ra LLM chọn 2 đoạn văn từ 2 chủ đề khác nhau (access control vs. leave policy) và gọi đó là "mâu thuẫn" — thực ra chỉ là 2 tài liệu độc lập.  
**Giải quyết:** Xác định đây là giới hạn của prompt hiện tại, đề xuất cải thiện bằng cách inject cặp thông tin mâu thuẫn thực sự vào context thay vì để LLM tự tìm.
