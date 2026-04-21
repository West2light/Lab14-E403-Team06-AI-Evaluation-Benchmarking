import json
import os


def validate_lab():
    print("🔍 Đang kiểm tra định dạng bài nộp...")

    required_files = [
        "reports/summary.json",
        "reports/benchmark_results.json",
        "analysis/failure_analysis.md"
    ]

    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"✅ Tìm thấy: {f}")
        else:
            print(f"❌ Thiếu file: {f}")
            missing.append(f)

    if missing:
        print(f"\n❌ Thiếu {len(missing)} file. Hãy bổ sung trước khi nộp bài.")
        return

    try:
        with open("reports/summary.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ File reports/summary.json không phải JSON hợp lệ: {e}")
        return

    if "metrics" not in data or "metadata" not in data:
        print("❌ File summary.json thiếu trường 'metrics' hoặc 'metadata'.")
        return

    metrics = data["metrics"]
    regression = data.get("regression", {})

    print(f"\n--- Thống kê nhanh ---")
    print(f"Tổng số cases: {data['metadata'].get('total', 'N/A')}")
    print(f"Điểm trung bình: {metrics.get('avg_score', 0):.2f}")
    print(f"Pass Rate: {metrics.get('pass_rate', 0) * 100:.1f}%")

    has_retrieval = "hit_rate" in metrics
    if has_retrieval:
        print(f"✅ Đã tìm thấy Retrieval Metrics (Hit Rate: {metrics['hit_rate']*100:.1f}%)")
    else:
        print("⚠️ CẢNH BÁO: Thiếu Retrieval Metrics (hit_rate).")

    has_mrr = "avg_mrr" in metrics
    if has_mrr:
        print(f"✅ Đã tìm thấy MRR (Avg MRR: {metrics['avg_mrr']:.4f})")
    else:
        print("⚠️ CẢNH BÁO: Thiếu Retrieval Metrics (avg_mrr).")

    has_multi_judge = "agreement_rate" in metrics
    if has_multi_judge:
        print(f"✅ Đã tìm thấy Multi-Judge Metrics (Agreement Rate: {metrics['agreement_rate']*100:.1f}%)")
    else:
        print("⚠️ CẢNH BÁO: Thiếu Multi-Judge Metrics (agreement_rate).")

    if "avg_latency" in metrics and "total_tokens" in metrics:
        print(
            f"✅ Đã tìm thấy Performance/Cost Metrics "
            f"(Avg Latency: {metrics['avg_latency']:.4f}s, Total Tokens: {metrics['total_tokens']})"
        )
    else:
        print("⚠️ CẢNH BÁO: Thiếu Performance/Cost Metrics (avg_latency hoặc total_tokens).")

    if "failed_cases" in metrics:
        print(f"✅ Đã tìm thấy Failure Metrics (Failed Cases: {metrics['failed_cases']})")
    else:
        print("⚠️ CẢNH BÁO: Thiếu Failure Metrics (failed_cases).")

    if data["metadata"].get("version"):
        print("✅ Đã tìm thấy thông tin phiên bản Agent (Regression Mode)")

    if regression.get("decision"):
        print(f"✅ Release Gate Decision: {regression['decision']}")
        gate_reasons = regression.get("gate_reasons") or []
        if gate_reasons:
            print("   Lý do gate:")
            for reason in gate_reasons:
                print(f"   - {reason}")

    print("\n🚀 Bài lab đã sẵn sàng để chấm điểm!")


if __name__ == "__main__":
    validate_lab()
