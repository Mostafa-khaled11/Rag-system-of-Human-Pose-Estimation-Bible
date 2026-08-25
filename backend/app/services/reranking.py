from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from app.core.vector_store import SearchHit

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class RerankResult:
    hits: list[SearchHit]
    applied: bool
    fallback: bool = False


class RerankingService:
    """A dependency-free CPU reranker combining lexical relevance and vector score."""

    def __init__(self, enabled: bool, model: str, vector_weight: float) -> None:
        if model != "lexical-v1":
            raise ValueError(f"Unsupported reranker model: {model}")
        self.enabled = enabled
        self.model = model
        self.vector_weight = vector_weight

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}

    def lexical_score(self, question: str, hit: SearchHit) -> float:
        query = self._tokens(question)
        document = self._tokens(str(hit.payload.get("chunk_text", "")))
        if not query or not document:
            return 0.0
        return len(query & document) / math.sqrt(len(query) * len(document))

    def score(self, question: str, hit: SearchHit) -> float:
        lexical = self.lexical_score(question, hit)
        vector = max(0.0, min(1.0, (hit.score + 1.0) / 2.0))
        return self.vector_weight * vector + (1 - self.vector_weight) * lexical

    def rerank(self, question: str, hits: list[SearchHit], final_k: int) -> RerankResult:
        if not self.enabled:
            return RerankResult(hits[:final_k], applied=False)
        ordered = sorted(hits, key=lambda hit: self.score(question, hit), reverse=True)
        return RerankResult(ordered[:final_k], applied=True)


class FallbackRerankingService:
    def __init__(self, primary: RerankingService | None) -> None:
        self.primary = primary

    def rerank(self, question: str, hits: list[SearchHit], final_k: int) -> RerankResult:
        if self.primary is None:
            return RerankResult(hits[:final_k], applied=False, fallback=True)
        try:
            return self.primary.rerank(question, hits, final_k)
        except Exception:
            logger.exception("reranker failed; using vector ordering")
            return RerankResult(hits[:final_k], applied=False, fallback=True)
