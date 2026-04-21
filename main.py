import asyncio
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List

from agent.main_agent import AgentV1, AgentV2
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


MIN_AGREEMENT_RATE = 0.5

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _tokens(text: str) -> set[str]:
    return set((text or "").lower().split())


def _overlap_score(reference: str, candidate: str) -> float:
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    if not reference_tokens:
        return 0.0
    return round(len(reference_tokens & candidate_tokens) / len(reference_tokens), 4)


class ExpertEvaluator:
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator(top_k=3)

    async def score(self, case: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
        expected_ids: List[str] = (
            case.get("ground_truth_doc_ids")
            or case.get("expected_retrieval_ids")
            or []
        )
        retrieved_ids: List[str] = (
            resp.get("retrieved_ids")
            or resp.get("metadata", {}).get("sources")
            or []
        )

        hit_rate = (
            self.retrieval_evaluator.calculate_hit_rate(expected_ids, retrieved_ids)
            if expected_ids
            else 0.0
        )
        mrr = (
            self.retrieval_evaluator.calculate_mrr(expected_ids, retrieved_ids)
            if expected_ids
            else 0.0
        )

        answer = resp.get("answer", "")
        context = " ".join(resp.get("contexts") or [])
        expected_answer = case.get("expected_answer", "")

        return {
            "faithfulness": _overlap_score(answer, context),
            "relevancy": _overlap_score(expected_answer, answer),
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr},
        }


def _safe_average(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _aggregate_results(agent_version: str, results: List[Dict[str, Any]], elapsed_seconds: float) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "metadata": {
                "version": agent_version,
                "total": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "metrics": {},
        }

    score_values = [float(result["judge"]["final_score"]) for result in results]
    agreement_values = [float(result["judge"].get("agreement_rate", 0.0)) for result in results]
    pass_count = sum(1 for result in results if result.get("status") == "pass")
    failed_cases = [result for result in results if result.get("status") == "error"]
    retry_count_total = sum(max(int(result.get("attempts", 1)) - 1, 0) for result in results)

    retrieval_results = [result for result in results if result.get("has_ground_truth")]
    hit_rate_values = [float(result["ragas"].get("hit_rate", 0.0)) for result in retrieval_results]
    mrr_values = [float(result["ragas"].get("mrr", 0.0)) for result in retrieval_results]

    latency_values = [float(result.get("latency", 0.0)) for result in results]
    token_values = [int(result.get("tokens_used", 0)) for result in results]

    backend_counts = Counter(result.get("retrieval_backend", "none") for result in results)
    judge_models = sorted(
        {
            model
            for result in results
            for model in result.get("judge", {}).get("judge_models", [])
        }
    )

    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": _safe_average(score_values),
            "pass_rate": round(pass_count / total, 4),
            "agreement_rate": _safe_average(agreement_values),
            "avg_hit_rate": _safe_average(hit_rate_values),
            "hit_rate": _safe_average(hit_rate_values),
            "avg_mrr": _safe_average(mrr_values),
            "retrieval_evaluated": len(retrieval_results),
            "failed_cases": len(failed_cases),
            "failure_rate": round(len(failed_cases) / total, 4),
            "retry_count_total": retry_count_total,
            "avg_latency": _safe_average(latency_values),
            "total_runtime": round(elapsed_seconds, 4),
            "total_tokens": sum(token_values),
            "avg_tokens": _safe_average([float(value) for value in token_values]),
            "judge_models": judge_models,
            "retrieval_backend_counts": dict(backend_counts),
            "retrieval_fallback_cases": sum(
                1 for result in results if result.get("retrieval_fallback_used")
            ),
        },
    }


def _build_gate(v1_summary: Dict[str, Any], v2_summary: Dict[str, Any]) -> Dict[str, Any]:
    v1_metrics = v1_summary["metrics"]
    v2_metrics = v2_summary["metrics"]
    gate_reasons: List[str] = []

    if v2_metrics["failed_cases"] > 0:
        gate_reasons.append("V2 has persistent benchmark errors after retry.")
    if v2_metrics["avg_score"] < v1_metrics["avg_score"]:
        gate_reasons.append("V2 average judge score regressed versus V1.")
    if v2_metrics["avg_hit_rate"] < v1_metrics["avg_hit_rate"]:
        gate_reasons.append("V2 retrieval hit rate regressed versus V1.")
    if v2_metrics["avg_mrr"] < v1_metrics["avg_mrr"]:
        gate_reasons.append("V2 retrieval MRR regressed versus V1.")
    if v2_metrics["agreement_rate"] < MIN_AGREEMENT_RATE:
        gate_reasons.append("V2 judge agreement is below the minimum threshold.")

    return {
        "decision": "APPROVE" if not gate_reasons else "BLOCK",
        "gate_reasons": gate_reasons,
    }


async def run_benchmark_with_results(agent_version: str, agent):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(agent, ExpertEvaluator(), LLMJudge())
    start_time = time.perf_counter()
    results = await runner.run_all(dataset)
    elapsed_seconds = time.perf_counter() - start_time
    summary = _aggregate_results(agent_version, results, elapsed_seconds)
    return results, summary


async def run_benchmark(version, agent):
    _, summary = await run_benchmark_with_results(version, agent)
    return summary


async def main():
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base", AgentV1())
    v2_results, v2_summary = await run_benchmark_with_results(
        "Agent_V2_Optimized", AgentV2()
    )

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    gate = _build_gate(v1_summary, v2_summary)
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"V2 Hit Rate: {v2_summary['metrics']['avg_hit_rate']}")
    print(f"V2 MRR: {v2_summary['metrics']['avg_mrr']}")
    print(f"V2 Agreement: {v2_summary['metrics']['agreement_rate']}")
    print(f"V2 Failed Cases: {v2_summary['metrics']['failed_cases']}")

    report_summary = {
        "metadata": {
            "total": v2_summary["metadata"]["total"],
            "version": v2_summary["metadata"]["version"],
            "timestamp": v2_summary["metadata"]["timestamp"],
            "versions_compared": ["V1", "V2"],
        },
        "metrics": v2_summary["metrics"],
        "v1_metrics": v1_summary["metrics"],
        "regression": {
            "v1": v1_summary["metrics"],
            "v2": v2_summary["metrics"],
            "delta": {
                "avg_score": round(delta, 4),
                "avg_hit_rate": round(
                    v2_summary["metrics"]["avg_hit_rate"] - v1_summary["metrics"]["avg_hit_rate"],
                    4,
                ),
                "avg_mrr": round(
                    v2_summary["metrics"]["avg_mrr"] - v1_summary["metrics"]["avg_mrr"],
                    4,
                ),
                "agreement_rate": round(
                    v2_summary["metrics"]["agreement_rate"] - v1_summary["metrics"]["agreement_rate"],
                    4,
                ),
                "avg_latency": round(
                    v2_summary["metrics"]["avg_latency"] - v1_summary["metrics"]["avg_latency"],
                    4,
                ),
                "failed_cases": v2_summary["metrics"]["failed_cases"] - v1_summary["metrics"]["failed_cases"],
            },
            **gate,
        },
    }

    report_results = {
        "v1": v1_results,
        "v2": v2_results,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(report_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report_results, f, ensure_ascii=False, indent=2)

    if gate["decision"] == "APPROVE":
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")
        for reason in gate["gate_reasons"]:
            print(f"- {reason}")


if __name__ == "__main__":
    asyncio.run(main())
