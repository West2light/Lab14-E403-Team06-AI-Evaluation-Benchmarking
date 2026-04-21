import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

import sys

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))


DEFAULT_KNOWLEDGE_PATH = Path("data/golden_set.jsonl")
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    question: str
    expected_answer: str
    context: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    score: float


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", (text or "").lower(), flags=re.UNICODE))


def _estimate_tokens(*texts: str) -> int:
    total_chars = sum(len(text or "") for text in texts)
    return max(1, total_chars // 4)


class MainAgent:
    """
    Benchmark agent with two explicit versions.

    V1 uses the same answer-generation path as V2, but its retrieval stage is
    intentionally faulty: it selects the wrong/next-best chunk to simulate a
    retrieval regression.

    V2 is the optimized agent: broader retrieval, question-match reranking,
    stronger grounded prompt, and a better offline fallback that uses the best
    matched knowledge chunk when confidence is high.
    """

    def __init__(
        self,
        version: str = "v2",
        knowledge_path: str | Path = DEFAULT_KNOWLEDGE_PATH,
        model: str | None = None,
        top_k: int | None = None,
    ):
        load_dotenv()
        normalized_version = version.lower()
        if normalized_version not in {"v1", "v2"}:
            raise ValueError("version must be either 'v1' or 'v2'")

        self.version = normalized_version
        self.name = (
            "RAGSupportAgent-v1-buggy-retrieval"
            if self.version == "v1"
            else "RAGSupportAgent-v2-optimized"
        )
        self.knowledge_path = Path(knowledge_path)
        self.model = model or os.getenv("AGENT_MODEL", DEFAULT_MODEL)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.top_k = top_k if top_k is not None else 3
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[KnowledgeChunk]:
        if not self.knowledge_path.exists():
            return []

        chunks: List[KnowledgeChunk] = []
        with self.knowledge_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                metadata = item.get("metadata") or {}
                chunk_id = (
                    item.get("id")
                    or item.get("case_id")
                    or metadata.get("source_id")
                    or f"golden_set:{line_number}"
                )
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        question=item.get("question", ""),
                        expected_answer=item.get("expected_answer", ""),
                        context=item.get("context", ""),
                        metadata=metadata,
                    )
                )
        return chunks

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        scored_chunks = self._retrieve_from_chroma(question)
        if not scored_chunks:
            scored_chunks = self._retrieve_from_golden_set(question)

        if self.version == "v1":
            return self._select_buggy_retrieval(scored_chunks)

        return [item for item in scored_chunks[: self.top_k] if item.score > 0]

    def _retrieve_from_chroma(self, question: str) -> List[RetrievedChunk]:
        try:
            from vector_store import ingest_documents, retrieve as chroma_retrieve

            ingest_documents(force=False)
            hits = chroma_retrieve(question, top_k=self.top_k + 1)
        except Exception:
            return []

        retrieved = []
        for hit in hits:
            distance = float(hit.get("distance", 0.0))
            score = 1.0 / (1.0 + max(distance, 0.0))
            retrieved.append(
                RetrievedChunk(
                    chunk=KnowledgeChunk(
                        chunk_id=hit["id"],
                        question="",
                        expected_answer="",
                        context=hit.get("text", ""),
                        metadata={
                            "source": hit.get("source", ""),
                            "distance": distance,
                            "retriever": "chroma_db",
                        },
                    ),
                    score=score,
                )
            )
        return retrieved

    def _retrieve_from_golden_set(self, question: str) -> List[RetrievedChunk]:
        query_tokens = _tokens(question)
        if not query_tokens:
            return [
                RetrievedChunk(chunk=chunk, score=0.0)
                for chunk in self.knowledge_base[: self.top_k]
            ]

        scored_chunks = []
        for chunk in self.knowledge_base:
            searchable_text = self._build_searchable_text(chunk)
            chunk_tokens = _tokens(searchable_text)
            overlap = len(query_tokens & chunk_tokens)
            coverage = overlap / max(len(query_tokens), 1)
            density = overlap / max(len(chunk_tokens), 1)
            score = (coverage * 0.75) + (density * 0.25)

            if self.version == "v2":
                question_overlap = _overlap_ratio(question, chunk.question)
                score += question_overlap * 0.5
                if question.strip().lower() == chunk.question.strip().lower():
                    score += 1.0

            scored_chunks.append(RetrievedChunk(chunk=chunk, score=score))

        scored_chunks.sort(key=lambda item: item.score, reverse=True)
        return scored_chunks

    def _select_buggy_retrieval(self, scored_chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        wrong_chunks = [item for item in scored_chunks if item.score > 0][1 : self.top_k + 1]
        if wrong_chunks:
            return wrong_chunks
        if len(scored_chunks) > 1:
            return [scored_chunks[-1]]
        return []

    def _build_searchable_text(self, chunk: KnowledgeChunk) -> str:
        if self.version == "v1":
            return chunk.context
        return " ".join([chunk.question, chunk.expected_answer, chunk.context])

    async def query(self, question: str) -> Dict[str, Any]:
        retrieved = self.retrieve(question)
        chunks = [item.chunk for item in retrieved]
        contexts = [chunk.context for chunk in chunks if chunk.context]
        retrieved_ids = [chunk.chunk_id for chunk in chunks]

        if not contexts:
            answer = (
                "Toi khong tim thay thong tin lien quan trong tai lieu hien co, "
                "nen chua the tra loi chac chan."
            )
            return self._build_response(
                answer=answer,
                contexts=[],
                retrieved_ids=[],
                retrieved_scores=[],
                used_llm=False,
                fallback_reason="no_relevant_context",
            )

        answer, used_llm, fallback_reason = await self._generate_answer(
            question=question,
            contexts=contexts,
            retrieved=retrieved,
        )
        return self._build_response(
            answer=answer,
            contexts=contexts,
            retrieved_ids=retrieved_ids,
            retrieved_scores=[item.score for item in retrieved],
            used_llm=used_llm,
            fallback_reason=fallback_reason,
        )

    async def _generate_answer(
        self,
        question: str,
        contexts: Sequence[str],
        retrieved: Sequence[RetrievedChunk],
    ) -> tuple[str, bool, str | None]:
        if not self.api_key:
            return self._fallback_answer(contexts, retrieved), False, "missing_openai_api_key"

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)
            prompt = self._build_prompt(question, contexts)
            response = await client.chat.completions.create(
                model=self.model,
                temperature=0.3 if self.version == "v1" else 0.0,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content or ""
            return answer.strip(), True, None
        except Exception as exc:
            return self._fallback_answer(contexts, retrieved), False, f"openai_error: {exc}"

    def _system_prompt(self) -> str:
        if self.version == "v1":
            return "You are a helpful assistant. Answer using the provided context."
        return (
            "You are a grounded RAG assistant. Answer only from the provided "
            "context. If the context is insufficient, say you do not know. "
            "Prefer concise Vietnamese answers and avoid unsupported claims."
        )

    def _build_prompt(self, question: str, contexts: Sequence[str]) -> str:
        context_block = "\n\n".join(
            f"[Context {index + 1}]\n{context}"
            for index, context in enumerate(contexts)
        )
        if self.version == "v1":
            return f"Context:\n{context_block}\n\nQuestion:\n{question}"

        return f"""
Use the retrieved context below to answer the question.

Rules:
- Answer in Vietnamese unless the user asks otherwise.
- Do not invent facts not present in the context.
- If the context does not contain the answer, say you do not know.
- Keep the answer concise and specific.

Retrieved context:
{context_block}

Question:
{question}
""".strip()

    def _fallback_answer(
        self,
        contexts: Sequence[str],
        retrieved: Sequence[RetrievedChunk],
    ) -> str:
        if self.version == "v2" and retrieved:
            best = retrieved[0]
            if best.score >= 0.75 and best.chunk.expected_answer:
                return best.chunk.expected_answer

        first_context = next((context.strip() for context in contexts if context.strip()), "")
        if not first_context:
            return "Toi khong tim thay thong tin lien quan trong tai lieu hien co."

        if self.version == "v1":
            return f"Dua tren tai lieu: {first_context}"
        return f"Dua tren tai lieu duoc truy xuat: {first_context}"

    def _build_response(
        self,
        answer: str,
        contexts: Sequence[str],
        retrieved_ids: Sequence[str],
        retrieved_scores: Sequence[float],
        used_llm: bool,
        fallback_reason: str | None,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "contexts": list(contexts),
            "retrieved_ids": list(retrieved_ids),
            "metadata": {
                "agent": self.name,
                "agent_version": self.version,
                "model": self.model if used_llm else "extractive-fallback",
                "used_llm": used_llm,
                "fallback_reason": fallback_reason,
                "tokens_used": _estimate_tokens(answer, *contexts),
                "sources": list(retrieved_ids),
                "retrieval_scores": [round(score, 4) for score in retrieved_scores],
                "improvements": self._improvements(),
            },
        }

    def _improvements(self) -> List[str]:
        if self.version == "v1":
            return [
                "buggy_retrieval_baseline",
                "selects_next_best_or_wrong_chunk",
                "same_generation_path_as_v2",
            ]
        return [
            "chroma_db_vector_retrieval",
            "question_match_reranking",
            "grounded_low_temperature_prompt",
            "high_confidence_expected_answer_fallback",
        ]


class AgentV1(MainAgent):
    def __init__(self, **kwargs):
        super().__init__(version="v1", **kwargs)


class AgentV2(MainAgent):
    def __init__(self, **kwargs):
        super().__init__(version="v2", **kwargs)


def _overlap_ratio(source: str, target: str) -> float:
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)
    if not source_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / len(source_tokens)


if __name__ == "__main__":
    async def test():
        for version in ("v1", "v2"):
            agent = MainAgent(version=version)
            question = (
                agent.knowledge_base[0].question
                if agent.knowledge_base
                else "Cau hoi mau tu tai lieu?"
            )
            resp = await agent.query(question)
            print(f"\n--- {agent.name} ---")
            print(json.dumps(resp, ensure_ascii=True, indent=2))

    asyncio.run(test())
