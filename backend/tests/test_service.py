import pytest

from app.core.config import Settings
from app.core.vector_store import DimensionMismatchError, SearchHit
from app.ingestion.chunking import PageText, config_fingerprint
from app.ingestion.pdf import ExtractedDocument
from app.retrieval.service import RagService


class FakeOllama:
    generated = False
    embedded = False

    async def embed(self, _model, texts):
        self.embedded = True
        return [[0.1, 0.2] for _ in texts]

    async def generate(self, *_args):
        self.generated = True
        return "answer"


class EmptyStore:
    async def search(self, *_args):
        return []


@pytest.mark.asyncio
async def test_empty_retrieval_skips_generation() -> None:
    ollama = FakeOllama()
    service = RagService(Settings(_env_file=None), ollama, EmptyStore())
    response = await service.query("What is an unrelated fact?", None, None)
    assert response.status == "insufficient_context"
    assert response.citations == []
    assert not ollama.generated


def test_citation_serialization() -> None:
    service = RagService(Settings(_env_file=None), FakeOllama(), EmptyStore())
    citation = service._citation(
        SearchHit("id", 0.73, {"page": 42, "chapter": "Pose", "chunk_text": "support"})
    )
    assert citation.model_dump() == {
        "page": 42,
        "chapter": "Pose",
        "chunk_id": "id",
        "score": 0.73,
        "excerpt": "support",
    }


class DimensionStore:
    async def collection_dimension(self):
        return 3

    async def search(self, vector, *_args):
        if len(vector) != 3:
            raise DimensionMismatchError("mismatch")
        return []


class EvidenceStore:
    async def search(self, *_args):
        return [
            SearchHit(
                "id",
                0.9,
                {
                    "record_type": "chunk",
                    "page": 7,
                    "chapter": "Pose",
                    "chunk_text": "Supporting evidence",
                    "source_filename": "book.pdf",
                },
            )
        ]


@pytest.mark.asyncio
async def test_answer_without_model_marker_gets_verified_page_marker() -> None:
    response = await RagService(Settings(_env_file=None), FakeOllama(), EvidenceStore()).query(
        "question", None, None
    )
    assert response.answer.endswith("Supporting pages: [p. 7]")


@pytest.mark.asyncio
async def test_dimension_mismatch() -> None:
    service = RagService(Settings(_env_file=None), FakeOllama(), DimensionStore())
    with pytest.raises(DimensionMismatchError):
        await service.query("question", None, None)


class ExistingStore:
    async def metadata(self):
        return {
            "document_id": "hash",
            "config_fingerprint": config_fingerprint(1200, 200, 200),
            "chunk_count": 3,
            "low_text_pages": [],
        }


@pytest.mark.asyncio
async def test_idempotent_ingestion_skips_embeddings(monkeypatch, tmp_path) -> None:
    source = tmp_path / "book.pdf"
    source.touch()
    settings = Settings(_env_file=None, book_path=source)
    document = ExtractedDocument("book.pdf", "hash", [PageText(1, "text")], [])
    monkeypatch.setattr("app.retrieval.service.extract_pdf", lambda *_: document)
    ollama = FakeOllama()
    response = await RagService(settings, ollama, ExistingStore()).ingest(False, None, None)
    assert response.status == "unchanged"
    assert response.chunk_count == 3
    assert not ollama.embedded
