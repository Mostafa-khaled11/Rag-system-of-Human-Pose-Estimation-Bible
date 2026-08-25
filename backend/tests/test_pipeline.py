import pytest

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.ollama import OllamaError
from app.core.vector_store import SearchHit
from app.services.embedding import EmbeddingService
from app.services.generation import GenerationService
from app.services.rag import RagService


def evidence(score: float = 0.9, page: int = 7) -> SearchHit:
    return SearchHit(
        "chunk",
        score,
        {
            "record_type": "chunk",
            "page": page,
            "chapter": "Pose",
            "chunk_text": "Monocular depth ambiguity has multiple possible 3D poses.",
            "source_filename": "book.pdf",
        },
    )


class Ollama:
    generated = False

    async def embed(self, _model, texts):
        return [[0.1, 0.2] for _ in texts]

    async def generate(self, *_args):
        self.generated = True
        return "Depth is ambiguous [C1], but this invalid marker is ignored [C99]."

    async def generate_stream(self, *_args):
        yield "Depth is "
        yield "ambiguous [C1]."


class Store:
    def __init__(self, hits):
        self.hits = hits
        self.limit = None

    async def search(self, _vector, limit, _threshold):
        self.limit = limit
        return self.hits


@pytest.mark.asyncio
async def test_answerable_pipeline_uses_candidates_and_only_retrieved_citations() -> None:
    store = Store([evidence()])
    service = RagService(Settings(_env_file=None, reranking_enabled=True), Ollama(), store)
    response = await service.query("Why is monocular depth ambiguous?", None, None, "req-1")
    assert store.limit == 20
    assert response.answerable
    assert response.request_id == "req-1"
    assert [citation.page for citation in response.citations] == [7]
    assert "C99" not in response.answer
    assert response.reranking_applied


@pytest.mark.asyncio
async def test_weak_retrieval_returns_insufficient_context_without_generation() -> None:
    ollama = Ollama()
    service = RagService(Settings(_env_file=None), ollama, Store([evidence(score=0.1)]))
    response = await service.query("unrelated question", None, None)
    assert response.insufficient_context
    assert not response.answerable
    assert response.citations == []
    assert not ollama.generated


@pytest.mark.asyncio
async def test_stream_has_distinguishable_event_types_and_final_metadata() -> None:
    service = RagService(
        Settings(_env_file=None, reranking_enabled=True), Ollama(), Store([evidence()])
    )
    events = [event async for event in service.stream_query("depth ambiguity", None, None, "r")]
    assert [event["type"] for event in events] == [
        "status",
        "retrieval",
        "token",
        "token",
        "final",
        "done",
    ]
    assert events[-2]["data"]["citations"][0]["page"] == 7


class FailingOllama:
    async def embed(self, *_args):
        raise OllamaError("offline")

    async def generate(self, *_args):
        raise TimeoutError


@pytest.mark.asyncio
async def test_ollama_unavailable_during_embedding_is_structured() -> None:
    with pytest.raises(AppError) as caught:
        await EmbeddingService(FailingOllama(), "embed").embed(["text"])
    assert caught.value.code == ErrorCode.OLLAMA_UNAVAILABLE
    assert caught.value.retryable


@pytest.mark.asyncio
async def test_generation_timeout_is_structured() -> None:
    with pytest.raises(AppError) as caught:
        await GenerationService(FailingOllama(), "model", 0.1).generate("q", "c")
    assert caught.value.code == ErrorCode.QUERY_TIMEOUT
    assert caught.value.status_code == 504
