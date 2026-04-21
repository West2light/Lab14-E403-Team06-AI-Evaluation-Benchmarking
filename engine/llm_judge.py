import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Sequence

from dotenv import load_dotenv


PASS_THRESHOLD = 3.5
CONFLICT_SCORE_GAP = 2.0


@dataclass(frozen=True)
class JudgeResult:
    judge_name: str
    model: str
    score: float
    passed: bool
    reason: str
    provider: str
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_name": self.judge_name,
            "provider": self.provider,
            "model": self.model,
            "score": round(self.score, 2),
            "pass": self.passed,
            "reason": self.reason,
            "used_fallback": self.used_fallback,
        }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", (text or "").lower(), flags=re.UNICODE))


def _overlap_ratio(source: str, target: str) -> float:
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)
    if not source_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / len(source_tokens)


def _clamp_score(score: float) -> float:
    return max(1.0, min(5.0, score))


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("empty judge response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_judge_payload(payload: Dict[str, Any]) -> tuple[float, bool, str]:
    raw_score = payload.get("score", 1)
    score = _clamp_score(float(raw_score))
    passed = bool(payload.get("pass", score >= PASS_THRESHOLD))
    reason = str(payload.get("reason", "No reason returned by judge.")).strip()
    return score, passed, reason


def _build_judge_prompt(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: Sequence[str] | None = None,
) -> str:
    context_block = "\n\n".join(contexts or [])
    return f"""
You are an impartial AI evaluation judge.

Grade the agent answer from 1 to 5 using this rubric:
5 = correct, complete, and fully grounded in the provided context.
4 = mostly correct with only minor missing details.
3 = partially correct but incomplete or weakly grounded.
2 = mostly incorrect or poorly supported.
1 = hallucinated, irrelevant, unsafe, or contradicts the context.

Return ONLY valid JSON with this exact schema:
{{
  "score": number,
  "pass": boolean,
  "reason": "short explanation"
}}

Question:
{question}

Expected answer / ground truth:
{ground_truth}

Retrieved context:
{context_block}

Agent answer:
{answer}
""".strip()


class BaseJudge:
    def __init__(self, judge_name: str, provider: str, model: str):
        self.judge_name = judge_name
        self.provider = provider
        self.model = model

    async def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: Sequence[str] | None = None,
    ) -> JudgeResult:
        raise NotImplementedError


class HeuristicJudge(BaseJudge):
    def __init__(self, judge_name: str, model: str, mode: str):
        super().__init__(judge_name, "local_fallback", model)
        self.mode = mode

    async def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: Sequence[str] | None = None,
    ) -> JudgeResult:
        await asyncio.sleep(0)
        if self.mode == "faithfulness":
            joined_context = " ".join(contexts or [])
            answer_to_context = _overlap_ratio(answer, joined_context)
            answer_to_truth = _overlap_ratio(answer, ground_truth)
            score = _clamp_score(1.0 + (answer_to_context * 2.4) + (answer_to_truth * 1.6))
            reason = "Fallback faithfulness score based on answer/context overlap."
        else:
            answer_to_truth = _overlap_ratio(ground_truth, answer)
            question_to_answer = _overlap_ratio(question, answer)
            score = _clamp_score(1.0 + (answer_to_truth * 3.2) + (question_to_answer * 0.8))
            reason = "Fallback accuracy score based on answer/ground-truth overlap."

        return JudgeResult(
            judge_name=self.judge_name,
            provider=self.provider,
            model=self.model,
            score=score,
            passed=score >= PASS_THRESHOLD,
            reason=reason,
            used_fallback=True,
        )


class OpenAIJudge(BaseJudge):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback: HeuristicJudge | None = None,
    ):
        super().__init__("openai_judge", "openai", model or os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.fallback = fallback or HeuristicJudge("openai_fallback_judge", self.model, "accuracy")

    async def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: Sequence[str] | None = None,
    ) -> JudgeResult:
        if not self.api_key:
            return await self.fallback.evaluate(question, answer, ground_truth, contexts)

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)
            prompt = _build_judge_prompt(question, answer, ground_truth, contexts)
            response = await client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict benchmark evaluator."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            score, passed, reason = _normalize_judge_payload(_extract_json_object(content))
            return JudgeResult(
                judge_name=self.judge_name,
                provider=self.provider,
                model=self.model,
                score=score,
                passed=passed,
                reason=reason,
            )
        except Exception as exc:
            fallback_result = await self.fallback.evaluate(question, answer, ground_truth, contexts)
            return JudgeResult(
                judge_name=self.judge_name,
                provider=self.provider,
                model=self.model,
                score=fallback_result.score,
                passed=fallback_result.passed,
                reason=f"OpenAI judge failed; used fallback. Error: {exc}",
                used_fallback=True,
            )


class GeminiJudge(BaseJudge):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback: HeuristicJudge | None = None,
    ):
        super().__init__("gemini_judge", "gemini", model or os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash"))
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.fallback = fallback or HeuristicJudge("gemini_fallback_judge", self.model, "faithfulness")

    async def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: Sequence[str] | None = None,
    ) -> JudgeResult:
        if not self.api_key:
            return await self.fallback.evaluate(question, answer, ground_truth, contexts)

        try:
            prompt = _build_judge_prompt(question, answer, ground_truth, contexts)
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                },
            }
            content = await asyncio.to_thread(self._call_gemini, payload)
            score, passed, reason = _normalize_judge_payload(_extract_json_object(content))
            return JudgeResult(
                judge_name=self.judge_name,
                provider=self.provider,
                model=self.model,
                score=score,
                passed=passed,
                reason=reason,
            )
        except Exception as exc:
            fallback_result = await self.fallback.evaluate(question, answer, ground_truth, contexts)
            return JudgeResult(
                judge_name=self.judge_name,
                provider=self.provider,
                model=self.model,
                score=fallback_result.score,
                passed=fallback_result.passed,
                reason=f"Gemini judge failed; used fallback. Error: {exc}",
                used_fallback=True,
            )

    def _call_gemini(self, payload: Dict[str, Any]) -> str:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {exc.code}: {error_body}") from exc

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)


class LLMJudge:
    def __init__(self, judges: Sequence[BaseJudge] | None = None):
        load_dotenv()
        self.judges = list(judges or [OpenAIJudge(), GeminiJudge()])
        self.rubrics = {
            "5": "Correct, complete, and fully grounded in context.",
            "4": "Mostly correct with minor missing details.",
            "3": "Partially correct but incomplete or weakly grounded.",
            "2": "Mostly incorrect or poorly supported.",
            "1": "Hallucinated, irrelevant, or contradicts the context.",
        }

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        judge_results = await asyncio.gather(
            *[
                judge.evaluate(question, answer, ground_truth, contexts)
                for judge in self.judges
            ]
        )

        scores = [result.score for result in judge_results]
        passes = [result.passed for result in judge_results]
        score_gap = max(scores) - min(scores) if scores else 0.0
        has_conflict = (len(set(passes)) > 1) or score_gap >= CONFLICT_SCORE_GAP
        agreement = 1.0 if not has_conflict and score_gap <= 1.0 else 0.0

        if has_conflict:
            final_score = min(scores)
            conflict_strategy = "conservative_min_score"
            reasoning = "Judges disagreed; final score uses the conservative minimum."
        else:
            final_score = sum(scores) / len(scores)
            conflict_strategy = "average_score"
            reasoning = "Judges agreed closely; final score uses the average."

        consensus_pass = final_score >= PASS_THRESHOLD
        individual_results = [result.to_dict() for result in judge_results]

        return {
            "final_score": round(final_score, 2),
            "pass": consensus_pass,
            "agreement_rate": agreement,
            "has_conflict": has_conflict,
            "conflict_strategy": conflict_strategy,
            "score_gap": round(score_gap, 2),
            "reasoning": reasoning,
            "individual_results": individual_results,
            "individual_scores": {
                f"{result.provider}:{result.model}": round(result.score, 2)
                for result in judge_results
            },
            "used_fallback": any(result.used_fallback for result in judge_results),
        }

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict[str, Any]:
        first_order = await self.evaluate_multi_judge(
            "Compare two candidate answers.", response_a, response_b, [response_b]
        )
        swapped_order = await self.evaluate_multi_judge(
            "Compare two candidate answers.", response_b, response_a, [response_a]
        )
        return {
            "first_order_score": first_order["final_score"],
            "swapped_order_score": swapped_order["final_score"],
            "score_delta": round(
                abs(first_order["final_score"] - swapped_order["final_score"]), 2
            ),
        }
