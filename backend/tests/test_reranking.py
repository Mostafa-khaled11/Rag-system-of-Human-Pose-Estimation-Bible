from app.core.vector_store import SearchHit
from app.services.reranking import FallbackRerankingService, RerankingService


def hit(identifier: str, score: float, text: str) -> SearchHit:
    return SearchHit(identifier, score, {"chunk_text": text})


def test_reranking_enabled_changes_order_by_query_relevance() -> None:
    service = RerankingService(True, "lexical-v1", vector_weight=0.2)
    result = service.rerank(
        "monocular depth ambiguity",
        [
            hit("vector-first", 0.95, "unrelated content"),
            hit("relevant", 0.7, "monocular depth ambiguity"),
        ],
        2,
    )
    assert [item.id for item in result.hits] == ["relevant", "vector-first"]
    assert result.applied


def test_reranking_disabled_preserves_vector_order() -> None:
    service = RerankingService(False, "lexical-v1", vector_weight=0.2)
    result = service.rerank(
        "monocular depth ambiguity",
        [hit("first", 0.9, "none"), hit("second", 0.8, "monocular depth ambiguity")],
        1,
    )
    assert [item.id for item in result.hits] == ["first"]
    assert not result.applied


def test_reranker_runtime_failure_falls_back_to_vector_order() -> None:
    class Broken:
        def rerank(self, *_args):
            raise RuntimeError("broken")

    result = FallbackRerankingService(Broken()).rerank(
        "question", [hit("first", 0.9, "a"), hit("second", 0.8, "b")], 1
    )
    assert [item.id for item in result.hits] == ["first"]
    assert result.fallback
    assert not result.applied


def test_reranker_initialization_failure_can_use_fallback() -> None:
    result = FallbackRerankingService(None).rerank("question", [hit("first", 0.9, "a")], 1)
    assert [item.id for item in result.hits] == ["first"]
    assert result.fallback
