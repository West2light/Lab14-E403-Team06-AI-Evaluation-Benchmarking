# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 66
- **So sánh Regression:** V1 `avg_score = 2.2247` → V2 `avg_score = 2.9686` (`+0.7439`)
- **Tỉ lệ Pass/Fail của V2:** 27 pass / 39 fail (`40.9%` pass rate)
- **Điểm Retrieval trung bình của V2** *(trên 39 case có ground truth)*:
  - Hit Rate: `0.6923`
  - MRR: `0.6496`
- **Điểm chất lượng trung bình của V2:**
  - Faithfulness: `0.5128`
  - Relevancy: `0.3238`
- **Điểm LLM-Judge trung bình của V2:** `2.9686 / 5.0`
- **Agreement Rate của 2 Judge:** `0.7268`
- **Hiệu năng:** `avg_latency = 12.8693s`, `total_runtime = 361.8452s`
- **Chi phí ước lượng:** `total_tokens = 20251`, `avg_tokens = 306.8333`
- **Release Gate:** `APPROVE`

## 2. Phân nhóm lỗi (Failure Clustering)

### Nhóm 1 — Retrieval miss trên câu factual ngắn, entity cụ thể
- **Số lượng ước tính:** 12 case retrieval miss (`hit_rate = 0`) trong các case có ground truth.
- **Biểu hiện:** Agent trả lời “Tôi không biết.” hoặc trả lời sai dù đáp án có tồn tại rõ trong tài liệu.
- **Ví dụ:**
  - `tc_007`: “Hệ thống quản lý danh tính nào được sử dụng?” → đáp án đúng là `Okta`, nhưng agent trả lời “Tôi không biết.”
  - `tc_015`: “Số điện thoại khẩn cấp ngoài giờ là gì?” → agent trả lời “Tôi không biết.”
  - `tc_020`: “Ngày hiệu lực của quy định xử lý sự cố là khi nào?” → agent trả lời “Tôi không biết.”
- **Nguyên nhân dự kiến:** embedding retrieval chưa ổn với câu hỏi ngắn mang tính lookup/entity; top-k hiện tại vẫn bỏ lỡ chunk chứa fact đích trong một số case.

### Nhóm 2 — Không xử lý tốt multi-turn và correction
- **Số lượng:** 6 case fail thuộc `multi_turn` và `multi_turn_correction`.
- **Biểu hiện:** câu hỏi lượt 2 phụ thuộc ngữ cảnh trước đó nhưng agent xử lý như một câu độc lập và trả lời “Tôi không biết.”
- **Ví dụ:**
  - `tc_050`: “Giờ làm việc của bộ phận hỗ trợ đó là khi nào?”
  - `tc_054`: “Thực ra tôi muốn hỏi quy trình này có hiệu lực từ khi nào?”
  - `tc_059`: “Thực ra, tôi đang hỏi về sản phẩm đã được kích hoạt, không phải hàng kỹ thuật số.”
- **Nguyên nhân dự kiến:** benchmark runner hiện gọi `agent.query()` theo từng câu độc lập, không mang theo hội thoại trước đó; agent hiện tại cũng chưa có memory state cho multi-turn.

### Nhóm 3 — Edge/conflicting/ambiguous cases bị từ chối quá sớm
- **Số lượng:** 5 case fail thuộc `edge_ambiguous` và `edge_conflicting`.
- **Biểu hiện:** agent trả lời “Tôi không biết.” trong khi kỳ vọng đúng phải là làm rõ ambiguity hoặc chỉ ra điểm mâu thuẫn giữa hai đoạn.
- **Ví dụ:**
  - `tc_042`: “Tôi có thể kết nối VPN không?”
  - `tc_051`: yêu cầu chỉ ra mâu thuẫn giữa policy access control và leave policy
  - `tc_053`: yêu cầu chỉ ra sự khác biệt giữa hai đoạn A/B
- **Nguyên nhân dự kiến:** prompt hiện thiên về grounded QA một lượt, nhưng chưa hướng dẫn rõ cách xử lý ambiguity/conflict analysis.

### Nhóm 4 — Adversarial / out-of-context handling còn chưa bám expected behavior
- **Số lượng:** 15 case fail thuộc `adversarial` và `edge_out_of_context`.
- **Biểu hiện:** agent thường trả lời “Không biết.” hoặc “Tôi không biết.”, trong khi expected answer cần một dạng từ chối hoặc giải thích có kiểm soát hơn.
- **Nguyên nhân dự kiến:** judge chấm dựa trên expected answer cụ thể; câu trả lời quá ngắn tuy an toàn nhưng không đủ informative nên vẫn bị fail.

### Nhóm 5 — Judge reliability bị ảnh hưởng bởi quota Gemini
- **Số lượng:** 66/66 case của cả V1 và V2 đều có `used_fallback = true` ở Gemini judge.
- **Biểu hiện:** Gemini thật không chấm được do `HTTP 429 / RESOURCE_EXHAUSTED`, nên pipeline dùng heuristic fallback thay thế.
- **Tác động:** agreement và final score vẫn tính được, nhưng độ tin cậy của multi-judge bị giảm vì một judge không phải model thật.
- **Nguyên nhân trực tiếp:** vượt monthly spending cap của Gemini project.

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất của V2)

### Case #1 — `tc_007`: “Hệ thống quản lý danh tính nào được sử dụng?”
1. **Symptom:** Agent trả lời “Tôi không biết.” thay vì `Okta`.
2. **Why 1:** Retriever không lấy được đúng chunk chứa thông tin `IAM system: Okta`.
3. **Why 2:** Câu hỏi rất ngắn và xoay quanh entity cụ thể, khiến semantic retrieval kém ổn định.
4. **Why 3:** Top-k retrieval hiện tối ưu khá tốt cho câu giải thích dài hơn là câu lookup fact ngắn.
5. **Why 4:** Chưa có lexical reranking / exact-term boosting cho entity như `Okta`, email, hotline, effective date.
6. **Root Cause:** Retrieval chưa đủ mạnh cho factual lookup ngắn, đặc biệt với entity/keyword cụ thể.

### Case #2 — `tc_050`: “Giờ làm việc của bộ phận hỗ trợ đó là khi nào?”
1. **Symptom:** Agent trả lời “Tôi không biết.” dù đây là câu multi-turn phụ thuộc ngữ cảnh trước.
2. **Why 1:** Runner gửi câu lượt 2 như một query độc lập.
3. **Why 2:** Agent không có conversation memory để nối “đó” về đúng thực thể ở lượt 1.
4. **Why 3:** Dataset có multi-turn cases nhưng pipeline inference hiện chưa mô phỏng hội thoại nhiều lượt thật.
5. **Why 4:** Benchmark hiện tối ưu cho single-turn RAG hơn là dialogue state tracking.
6. **Root Cause:** Thiết kế runner/agent chưa hỗ trợ context carry-over cho multi-turn evaluation.

### Case #3 — `tc_053`: case conflicting documents
1. **Symptom:** Agent trả lời “Tôi không biết.” thay vì chỉ ra khác biệt giữa hai đoạn A/B.
2. **Why 1:** Agent hiện được prompt để trả lời ngắn gọn, grounded, và nói không biết khi context không đủ rõ.
3. **Why 2:** Khi gặp hai đoạn có nội dung khác nhau, agent thiếu chỉ dẫn rõ ràng để so sánh và tổng hợp mâu thuẫn.
4. **Why 3:** Retrieval pipeline lấy được context nhưng generation prompt không khuyến khích thao tác “compare/contrast”.
5. **Why 4:** Hệ thống đang tối ưu factual QA trước, chưa tối ưu analytical QA trên nhiều chunk.
6. **Root Cause:** Prompting/generation chưa hỗ trợ tốt cho conflict resolution tasks dù retrieval đã có thể lấy được dữ liệu liên quan.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] **Tăng chất lượng retrieval cho factual lookup:** thêm lexical bonus / keyword exact match cho email, hotline, product type, effective date, tên hệ thống.
- [ ] **Hỗ trợ multi-turn benchmark đúng nghĩa:** mở rộng runner để truyền lịch sử hội thoại cho `multi_turn` và `multi_turn_correction` cases.
- [ ] **Nâng prompt cho ambiguous/conflicting cases:** yêu cầu agent ưu tiên hỏi lại khi mơ hồ và nêu rõ cả hai phía khi phát hiện mâu thuẫn.
- [ ] **Tách policy trả lời an toàn khỏi policy trả lời “không biết”:** với adversarial/OOC nên trả lời an toàn nhưng giàu thông tin hơn expected answer hiện tại.
- [ ] **Khôi phục Gemini judge thật:** nạp lại quota/API budget để multi-judge consensus phản ánh đúng 2 model judge thay vì 1 model + heuristic fallback.
- [ ] **Phân tích thêm 12 retrieval miss cases:** lập danh sách riêng các câu hỏi factual có `hit_rate = 0` để tối ưu retrieval theo pattern câu hỏi ngắn.
