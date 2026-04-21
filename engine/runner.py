import asyncio
import time
from typing import Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        ragas_scores = await self.evaluator.score(test_case, response)

        judge_contexts = list(response.get("contexts") or [])
        if test_case.get("context"):
            judge_contexts.append(test_case["context"])

        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
            judge_contexts,
        )

        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "ragas": {
                "hit_rate": ragas_scores["retrieval"]["hit_rate"],
                "mrr": ragas_scores["retrieval"]["mrr"],
                "faithfulness": ragas_scores["faithfulness"],
                "relevancy": ragas_scores["relevancy"],
            },
            "judge": {
                "final_score": judge_result["final_score"],
                "agreement_rate": judge_result["agreement_rate"],
                "individual_results": self._format_individual_judges(judge_result),
                "status": "conflict" if judge_result["has_conflict"] else "consensus",
            },
            "status": "pass" if judge_result["pass"] else "fail",
        }

    def _format_judge_reasoning(self, judge_result: Dict) -> str:
        individual_results = judge_result.get("individual_results") or []
        if not individual_results:
            return judge_result["reasoning"]

        reasons = []
        for result in individual_results:
            provider = result.get("provider", "judge")
            model = result.get("model", "unknown-model")
            score = result.get("score", "N/A")
            reason = result.get("reason", "")
            reasons.append(f"{provider} ({model}) score {score}: {reason}")
        return " | ".join(reasons)

    def _format_individual_judges(self, judge_result: Dict) -> Dict:
        formatted = {}
        for result in judge_result.get("individual_results") or []:
            model = result.get("model", "unknown-model")
            formatted[model] = {
                "score": result.get("score", 0),
                "reasoning": result.get("reason", ""),
            }
        return formatted

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
