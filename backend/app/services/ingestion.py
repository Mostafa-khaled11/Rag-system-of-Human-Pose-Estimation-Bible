from __future__ import annotations

import uuid

from qdrant_client import models

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, invalid_request
from app.core.vector_store import VectorStore
from app.ingestion.chunking import chunk_pages, config_fingerprint
from app.ingestion.pdf import SourceDocumentError, extract_pdf
from app.schemas.models import IngestResponse
from app.services.embedding import EmbeddingService


class IngestionService:
    def __init__(self, settings: Settings, embedding: EmbeddingService, store: VectorStore) -> None:
        self.settings = settings
        self.embedding = embedding
        self.store = store

    async def ingest(
        self, force: bool, chunk_size: int | None, chunk_overlap: int | None
    ) -> IngestResponse:
        cfg = self.settings.model_copy(
            update={
                "chunk_size": chunk_size or self.settings.chunk_size,
                "chunk_overlap": (
                    chunk_overlap if chunk_overlap is not None else self.settings.chunk_overlap
                ),
            }
        )
        try:
            cfg = Settings.model_validate(cfg.model_dump())
        except ValueError as exc:
            raise invalid_request(str(exc)) from exc
        try:
            document = extract_pdf(cfg.book_path, cfg.max_source_bytes)
        except (FileNotFoundError, SourceDocumentError) as exc:
            raise invalid_request(str(exc)) from exc
        fingerprint = config_fingerprint(cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_size)
        try:
            existing = await self.store.metadata()
        except Exception as exc:
            raise AppError(
                ErrorCode.QDRANT_UNAVAILABLE,
                "The vector store is unavailable.",
                status_code=503,
                retryable=True,
            ) from exc
        if (
            existing
            and existing.get("document_id") == document.sha256
            and existing.get("config_fingerprint") == fingerprint
            and not force
        ):
            return IngestResponse(
                status="unchanged",
                document_id=document.sha256,
                source_filename=document.source_filename,
                page_count=len(document.pages),
                chunk_count=int(existing.get("chunk_count", 0)),
                low_text_pages=list(existing.get("low_text_pages", [])),
                config_fingerprint=fingerprint,
            )
        chunks = chunk_pages(
            document.pages,
            document.sha256,
            fingerprint,
            cfg.chunk_size,
            cfg.chunk_overlap,
            cfg.min_chunk_size,
        )
        if not chunks:
            raise invalid_request("No indexable chunks were extracted")
        first_vectors = await self.embedding.embed([chunks[0].text])
        dimension = len(first_vectors[0])
        temporary = f"{cfg.collection_name}__building_{fingerprint}"
        try:
            current_dimension = await self.store.collection_dimension()
            if current_dimension and current_dimension != dimension and not force:
                raise AppError(
                    ErrorCode.INGESTION_FAILED,
                    "The existing index uses a different embedding dimension; "
                    "retry with force=true.",
                    status_code=409,
                )
            await self.store.delete(temporary)
            await self.store.create(temporary, dimension)
            for start in range(0, len(chunks), cfg.embedding_batch_size):
                batch = chunks[start : start + cfg.embedding_batch_size]
                vectors = (
                    first_vectors
                    if start == 0 and len(batch) == 1
                    else await self.embedding.embed([chunk.text for chunk in batch])
                )
                if any(len(vector) != dimension for vector in vectors):
                    raise AppError(
                        ErrorCode.EMBEDDING_FAILED,
                        "Embedding dimension changed during ingestion.",
                    )
                await self.store.upsert(
                    temporary,
                    [
                        models.PointStruct(
                            id=chunk.id,
                            vector=vector,
                            payload={
                                "record_type": "chunk",
                                "document_id": document.sha256,
                                "source_filename": document.source_filename,
                                "page": chunk.page,
                                "chapter": chunk.chapter,
                                "chunk_index": chunk.index,
                                "chunk_text": chunk.text,
                                "config_fingerprint": fingerprint,
                                "embedding_model": cfg.embedding_model,
                            },
                        )
                        for chunk, vector in zip(batch, vectors, strict=True)
                    ],
                )
            manifest_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"manifest:{document.sha256}:{fingerprint}")
            )
            await self.store.upsert(
                temporary,
                [
                    models.PointStruct(
                        id=manifest_id,
                        vector=[0.0] * dimension,
                        payload={
                            "record_type": "manifest",
                            "document_id": document.sha256,
                            "source_filename": document.source_filename,
                            "page_count": len(document.pages),
                            "chunk_count": len(chunks),
                            "low_text_pages": document.low_text_pages,
                            "config_fingerprint": fingerprint,
                            "embedding_model": cfg.embedding_model,
                        },
                    )
                ],
            )
            await self.store.activate(temporary)
        except AppError:
            await self.store.delete(temporary)
            raise
        except Exception as exc:
            await self.store.delete(temporary)
            raise AppError(
                ErrorCode.INGESTION_FAILED,
                "Book ingestion failed; the active index was left unchanged.",
                status_code=503,
                retryable=True,
            ) from exc
        return IngestResponse(
            status="indexed",
            document_id=document.sha256,
            source_filename=document.source_filename,
            page_count=len(document.pages),
            chunk_count=len(chunks),
            low_text_pages=document.low_text_pages,
            config_fingerprint=fingerprint,
        )
