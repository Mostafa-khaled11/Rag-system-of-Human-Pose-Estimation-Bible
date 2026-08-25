import pytest

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.retrieval.service import RagService


class Ollama:
    async def embed(self, _model, texts):
        return [[0.1, 0.2] for _ in texts]


class MissingIndex:
    async def exists(self):
        return False


class UnavailableStore:
    async def exists(self):
        raise RuntimeError("qdrant offline")


@pytest.mark.asyncio
async def test_missing_index_has_structured_not_ready_error() -> None:
    with pytest.raises(AppError) as caught:
        await RagService(Settings(_env_file=None), Ollama(), MissingIndex()).query(
            "human pose question", None, None
        )
    assert caught.value.code == ErrorCode.INDEX_NOT_READY
    assert caught.value.status_code == 409


@pytest.mark.asyncio
async def test_qdrant_failure_has_structured_retrieval_error() -> None:
    with pytest.raises(AppError) as caught:
        await RagService(Settings(_env_file=None), Ollama(), UnavailableStore()).query(
            "human pose question", None, None
        )
    assert caught.value.code == ErrorCode.RETRIEVAL_FAILED
    assert caught.value.retryable
