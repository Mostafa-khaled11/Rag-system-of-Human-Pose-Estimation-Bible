from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    force: bool = False
    chunk_size: int | None = Field(default=None, ge=200, le=8000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=2000)


class IngestResponse(BaseModel):
    status: Literal["indexed", "unchanged"]
    document_id: str
    source_filename: str
    page_count: int
    chunk_count: int
    low_text_pages: list[int]
    config_fingerprint: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=25)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class EvaluationRetrievalRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    rerank: bool = True
    top_k: int | None = Field(default=None, ge=1, le=25)


class Citation(BaseModel):
    page: int
    chapter: str | None = None
    chunk_id: str
    score: float
    excerpt: str


class RetrievedChunk(Citation):
    source_filename: str


class Timing(BaseModel):
    embedding_ms: float = 0
    retrieval_ms: float = 0
    reranking_ms: float = 0
    generation_ms: float = 0
    total_ms: float = 0


class QueryResponse(BaseModel):
    status: Literal["grounded", "insufficient_context"]
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    generation_model: str
    embedding_model: str
    timing: Timing
    request_id: str = ""
    answerable: bool = True
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    insufficient_context: bool = False
    reranking_applied: bool = False
    reranker_model: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool
    request_id: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ComponentHealth(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "ready", "degraded"]
    backend: ComponentHealth
    ollama: ComponentHealth
    qdrant: ComponentHealth
    index: ComponentHealth


class DocumentStatus(BaseModel):
    indexed: bool
    metadata: dict[str, Any] | None = None
