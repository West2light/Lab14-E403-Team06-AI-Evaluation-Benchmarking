import asyncio
import json
import os
import time

from agent.main_agent import AgentV1, AgentV2, MainAgent
from engine.llm_judge import LLMJudge
from engine.runner import BenchmarkRunner


class ExpertEvaluator:
    async def score(self, case, resp):
        return {
            "faithfulness": 0.9,
            "relevancy": 0.8,
            "retrieval": {"hit_rate": 1.0, "mrr": 0.5},
        }


async def run_benchmark_with_results(agent_version: str, agent: MainAgent):
    print(f"Starting benchmark for {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("Missing data/golden_set.jsonl. Run 'python data/synthetic_gen.py' first.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("data/golden_set.jsonl is empty. Add at least 1 test case.")
        return None, None

    runner = BenchmarkRunner(agent, ExpertEvaluator(), LLMJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    agreement_rate = sum(r["judge"]["agreement_rate"] for r in results) / total
    conflict_rate = sum(1 for r in results if r["judge"]["has_conflict"]) / total
    pass_rate = sum(1 for r in results if r["status"] == "pass") / total

    summary = {
        "metadata": {
            "version": agent_version,
            "agent_name": agent.name,
            "agent_version": agent.version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": avg_score,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results)
            / total,
            "agreement_rate": agreement_rate,
            "conflict_rate": conflict_rate,
            "pass_rate": pass_rate,
        },
        "multi_judge": {
            "num_judges": len(runner.judge.judges),
            "judge_models": [
                {"provider": judge.provider, "model": judge.model}
                for judge in runner.judge.judges
            ],
            "average_consensus_score": avg_score,
            "agreement_rate": agreement_rate,
            "conflict_rate": conflict_rate,
            "pass_rate": pass_rate,
            "used_fallback": any(r["judge"]["used_fallback"] for r in results),
            "conflict_policy": (
                "Use the minimum score when judges disagree by pass/fail "
                "or by >= 2 points."
            ),
        },
    }
    return results, summary


async def run_benchmark(version, agent):
    _, summary = await run_benchmark_with_results(version, agent)
    return summary


async def main():
    v1_agent = AgentV1()
    v2_agent = AgentV2()

    v1_summary = await run_benchmark("Agent_V1_Base", v1_agent)
    v2_results, v2_summary = await run_benchmark_with_results(
        "Agent_V2_Optimized", v2_agent
    )

    if not v1_summary or not v2_summary:
        print("Benchmark failed. Check data/golden_set.jsonl.")
        return

    print("\n--- Regression comparison ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.2f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.2f}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    v2_summary["regression"] = {
        "baseline_version": v1_summary["metadata"]["version"],
        "candidate_version": v2_summary["metadata"]["version"],
        "baseline_score": v1_summary["metrics"]["avg_score"],
        "candidate_score": v2_summary["metrics"]["avg_score"],
        "delta_score": delta,
        "decision": "APPROVE" if delta > 0 else "BLOCK_RELEASE",
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0:
        print("Decision: APPROVE")
    else:
        print("Decision: BLOCK RELEASE")


if __name__ == "__main__":
    asyncio.run(main())
