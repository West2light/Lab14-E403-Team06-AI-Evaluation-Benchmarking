from typing import List, Dict
import sys
import os

# Allow importing vector_store from sibling data/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Returns 1.0 if at least one expected_id is found in the top_k retrieved results, else 0.0.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank: 1 / (1-indexed position of first relevant result), or 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Run retrieval evaluation over the full golden dataset.

        Each record must have:
            - "question": str
            - "ground_truth_doc_ids": List[str]  (may be empty for adversarial/OOC cases)

        Returns:
            {
                "avg_hit_rate": float,
                "avg_mrr": float,
                "total": int,
                "evaluated": int,   # cases that had ground_truth_doc_ids
                "details": [{"id", "question", "hit_rate", "mrr", "retrieved_ids"}, ...]
            }
        """
        from vector_store import retrieve  # lazy import — requires ChromaDB to be ingested

        hit_rates: List[float] = []
        mrr_scores: List[float] = []
        details: List[Dict] = []

        for record in dataset:
            question = record.get("question", "")
            expected_ids: List[str] = record.get("ground_truth_doc_ids", [])

            # Skip cases that have no ground truth (adversarial / OOC)
            if not expected_ids:
                continue

            hits = retrieve(question, top_k=self.top_k)
            retrieved_ids = [h["id"] for h in hits]

            hr = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=self.top_k)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)

            hit_rates.append(hr)
            mrr_scores.append(mrr)
            details.append({
                "id": record.get("id", ""),
                "question": question,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
                "hit_rate": hr,
                "mrr": mrr,
            })

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

        return {
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "total": len(dataset),
            "evaluated": len(hit_rates),
            "details": details,
        }
