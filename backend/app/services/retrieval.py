from __future__ import annotations

from app.core.errors import AppError, ErrorCode
from app.core.vector_store import DimensionMismatchError, SearchHit, VectorStore


class RetrievalService:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    async def retrieve(self, vector: list[float], limit: int, threshold: float) -> list[SearchHit]:
        try:
            if hasattr(self.store, "exists") and not await self.store.exists():
                raise AppError(
                    ErrorCode.INDEX_NOT_READY,
                    "The book index is not ready.",
                    status_code=409,
                )
            hits = await self.store.search(vector, limit, threshold)
        except DimensionMismatchError as exc:
            raise AppError(
                ErrorCode.INDEX_NOT_READY,
                "The index is incompatible with the configured embedding model.",
                status_code=409,
            ) from exc
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                ErrorCode.RETRIEVAL_FAILED,
                "Passage retrieval failed.",
                status_code=503,
                retryable=True,
            ) from exc
        return [
            hit
            for hit in hits
            if hit.payload.get("record_type") == "chunk" and hit.payload.get("chunk_text")
        ]
