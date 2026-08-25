from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def normalized_words(text: str) -> set[str]:
    return {word.lower() for word in WORD_RE.findall(text)}


def key_point_coverage(answer: str, points: list[str]) -> float:
    if not points:
        return 1.0
    answer_words = normalized_words(answer)
    covered = 0
    for point in points:
        point_words = normalized_words(point)
        if point_words and len(answer_words & point_words) / len(point_words) >= 0.55:
            covered += 1
    return covered / len(points)


def reciprocal_rank(retrieved_pages: list[int], expected_pages: list[int]) -> float:
    expected = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            return 1 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalScores:
    hit_rate: float
    recall: float
    mrr: float


def retrieval_scores(results: list[dict[str, Any]], key: str) -> RetrievalScores:
    answerable = [item for item in results if item["sample"]["answerable_from_book"]]
    if not answerable:
        return RetrievalScores(0, 0, 0)
    hits = recalls = ranks = 0.0
    for item in answerable:
        expected = item["sample"]["expected_pages"]
        retrieved = item[key]
        overlap = set(expected) & set(retrieved)
        hits += bool(overlap)
        recalls += len(overlap) / len(set(expected)) if expected else 0
        ranks += reciprocal_rank(retrieved, expected)
    count = len(answerable)
    return RetrievalScores(hits / count, recalls / count, ranks / count)


def aggregate_generation(results: list[dict[str, Any]]) -> dict[str, float]:
    generated = [item for item in results if "response" in item]
    answerable = [item for item in generated if item["sample"]["answerable_from_book"]]
    unsupported = [item for item in generated if not item["sample"]["answerable_from_book"]]
    coverage = sum(item["key_point_coverage"] for item in answerable)
    citation_presence = sum(bool(item["citation_pages"]) for item in answerable)
    citation_correct = sum(item["citation_page_correct"] for item in answerable)
    grounded = sum(item["citations_from_retrieval"] for item in answerable)
    rejection = sum(item["unsupported_rejected"] for item in unsupported)
    return {
        "key_point_coverage": coverage / len(answerable) if answerable else 0,
        "citation_presence": citation_presence / len(answerable) if answerable else 0,
        "citation_page_correctness": citation_correct / len(answerable) if answerable else 0,
        "citations_from_retrieval": grounded / len(answerable) if answerable else 0,
        "unsupported_rejection_rate": rejection / len(unsupported) if unsupported else 0,
    }
