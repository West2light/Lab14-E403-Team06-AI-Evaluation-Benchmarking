import asyncio
import time
from typing import Any, Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    def _validate_response(self, response: Any) -> str | None:
        if not isinstance(response, dict):
            return "Agent did not return a dictionary response."

        if "answer" not in response:
            return "Agent response is missing the 'answer' field."

        answer = response.get("answer")
        if not isinstance(answer, str):
            return "Agent response field 'answer' must be a string."

        if not answer.strip():
            return "Agent returned an empty answer."

        return None

    async def _query_with_retry(self, question: str) -> tuple[Dict[str, Any] | None, int, str | None, str | None]:
        last_failure_reason: str | None = None
        last_error_type: str | None = None

        for attempt in range(1, 3):
            try:
                response = await self.agent.query(question)
                validation_error = self._validate_response(response)
                if validation_error is None:
                    return response, attempt, None, None

                last_failure_reason = validation_error
                last_error_type = "invalid_response"
            except Exception as exc:
                last_failure_reason = str(exc) or exc.__class__.__name__
                last_error_type = exc.__class__.__name__

            if attempt == 1:
                await asyncio.sleep(0)

        return None, 2, last_failure_reason, last_error_type

    def _base_result_fields(
        self,
        test_case: Dict[str, Any],
        latency: float,
        attempts: int,
        response: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata = response.get("metadata", {}) if isinstance(response, dict) else {}
        case_metadata = test_case.get("metadata") or {}

        return {
            "case_id": test_case.get("id", ""),
            "case_type": case_metadata.get("type", "unknown"),
            "has_ground_truth": bool(test_case.get("ground_truth_doc_ids")),
            "test_case": test_case["question"],
            "latency": latency,
            "attempts": attempts,
            "tokens_used": int(metadata.get("tokens_used") or 0),
            "retrieved_ids": list(response.get("retrieved_ids") or []) if isinstance(response, dict) else [],
            "retrieval_backend": metadata.get("retrieval_backend", "none"),
            "retrieval_fallback_used": bool(metadata.get("retrieval_fallback_used", False)),
            "agent_metadata": metadata,
        }

    def _format_individual_judges(self, judge_result: Dict[str, Any]) -> Dict[str, Any]:
        formatted = {}
        for result in judge_result.get("individual_results") or []:
            model = result.get("model", "unknown-model")
            formatted[model] = {
                "score": result.get("score", 0),
                "reasoning": result.get("reason", ""),
                "provider": result.get("provider", "judge"),
                "used_fallback": result.get("used_fallback", False),
            }
        return formatted

    def _build_error_result(
        self,
        test_case: Dict[str, Any],
        latency: float,
        attempts: int,
        failure_reason: str,
        error_type: str,
        response: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        result = self._base_result_fields(test_case, latency, attempts, response)
        result.update(
            {
                "agent_response": response.get("answer", "") if isinstance(response, dict) else "",
                "failure_reason": failure_reason,
                "error_type": error_type,
                "ragas": {
                    "hit_rate": 0.0,
                    "mrr": 0.0,
                    "faithfulness": 0.0,
                    "relevancy": 0.0,
                },
                "judge": {
                    "final_score": 0.0,
                    "agreement_rate": 0.0,
                    "individual_results": {},
                    "status": "error",
                    "reasoning": failure_reason,
                    "judge_models": [],
                    "used_fallback": False,
                    "has_conflict": False,
                    "score_gap": 0.0,
                },
                "status": "error",
            }
        )
        return result

    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        response, attempts, failure_reason, error_type = await self._query_with_retry(
            test_case["question"]
        )
        latency = time.perf_counter() - start_time

        if response is None:
            return self._build_error_result(
                test_case=test_case,
                latency=latency,
                attempts=attempts,
                failure_reason=failure_reason or "Agent query failed after retry.",
                error_type=error_type or "unknown_error",
            )

        try:
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
        except Exception as exc:
            return self._build_error_result(
                test_case=test_case,
                latency=latency,
                attempts=attempts,
                failure_reason=f"Scoring pipeline failed: {exc}",
                error_type=exc.__class__.__name__,
                response=response,
            )

        result = self._base_result_fields(test_case, latency, attempts, response)
        result.update(
            {
                "agent_response": response["answer"],
                "failure_reason": None,
                "error_type": None,
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
                    "reasoning": judge_result["reasoning"],
                    "judge_models": sorted(judge_result["individual_scores"].keys()),
                    "used_fallback": judge_result["used_fallback"],
                    "has_conflict": judge_result["has_conflict"],
                    "score_gap": judge_result["score_gap"],
                },
                "status": "pass" if judge_result["pass"] else "fail",
            }
        )
        return result

    async def run_all(self, dataset: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
