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
            "case_id": test_case.get("id") or test_case.get("case_id") or test_case["question"],
            "test_case": test_case["question"],
            "expected_answer": test_case["expected_answer"],
            "agent_response": response["answer"],
            "retrieved_contexts": response.get("contexts", []),
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "pass" if judge_result["pass"] else "fail",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
