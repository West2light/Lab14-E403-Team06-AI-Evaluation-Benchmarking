import asyncio
import json
import os
import time
from typing import Dict, List

from agent.main_agent import AgentV1, AgentV2
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


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

    async def score(self, case: Dict, resp: Dict) -> Dict:
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
    results = await runner.run_all(dataset)

    total = len(results)
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    hit_rate = sum(r["ragas"]["hit_rate"] for r in results) / total
    agreement_rate = sum(r["judge"]["agreement_rate"] for r in results) / total

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": avg_score,
            "hit_rate": hit_rate,
            "agreement_rate": agreement_rate,
        },
    }
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

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    report_summary = {
        "metadata": {
            "total": v2_summary["metadata"]["total"],
            "version": v2_summary["metadata"]["version"],
            "timestamp": v2_summary["metadata"]["timestamp"],
            "versions_compared": ["V1", "V2"],
        },
        "metrics": {
            "avg_score": v2_summary["metrics"]["avg_score"],
            "hit_rate": v2_summary["metrics"]["hit_rate"],
            "agreement_rate": v2_summary["metrics"]["agreement_rate"],
        },
        "regression": {
            "v1": {
                "score": v1_summary["metrics"]["avg_score"],
                "hit_rate": v1_summary["metrics"]["hit_rate"],
                "judge_agreement": v1_summary["metrics"]["agreement_rate"],
            },
            "v2": {
                "score": v2_summary["metrics"]["avg_score"],
                "hit_rate": v2_summary["metrics"]["hit_rate"],
                "judge_agreement": v2_summary["metrics"]["agreement_rate"],
            },
            "decision": "APPROVE" if delta > 0 else "BLOCK",
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

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
