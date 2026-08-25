from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import configure_logging
from app.core.metrics import metrics
from app.core.ollama import OllamaClient, OllamaError
from app.core.vector_store import VectorStore
from app.retrieval.service import RagService
from app.schemas.models import (
    ComponentHealth,
    DocumentStatus,
    EvaluationRetrievalRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

configure_logging(get_settings().log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ollama = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_timeout_seconds,
        settings.ollama_connect_timeout_seconds,
        settings.retry_count,
        settings.retry_backoff_seconds,
    )
    store = VectorStore(
        settings.qdrant_url, settings.collection_name, settings.qdrant_timeout_seconds
    )
    app.state.rag = RagService(settings, ollama, store)
    yield
    await ollama.close()
    await store.close()


app = FastAPI(title="Human Pose Estimation RAG API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


def service(request: Request) -> RagService:
    return request.app.state.rag


def request_id(request: Request) -> str:
    return str(request.state.request_id)


def error_payload(error: AppError, request_id_value: str) -> dict[str, object]:
    return {
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
            "request_id": request_id_value,
            "details": error.details,
        }
    }


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request.state.request_id = supplied[:128] if supplied else str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning(
        "request failed",
        extra={"request_id": request_id(request), "error_code": exc.code},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(exc, request_id(request)),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    error = AppError(
        ErrorCode.INVALID_REQUEST,
        "The request is invalid.",
        status_code=422,
        details={"errors": exc.errors()},
    )
    return JSONResponse(status_code=422, content=error_payload(error, request_id(request)))


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unexpected request failure", extra={"request_id": request_id(request)})
    error = AppError(
        ErrorCode.GENERATION_FAILED,
        "The request failed unexpectedly.",
        status_code=500,
        retryable=True,
    )
    return JSONResponse(status_code=500, content=error_payload(error, request_id(request)))


@app.get("/health")
async def health() -> dict[str, object]:
    """Cheap liveness probe: no network or model calls."""
    return {"status": "ok", "backend": {"ok": True, "detail": "running"}}


@app.get("/ready", response_model=HealthResponse)
async def ready(rag: RagService = Depends(service)) -> HealthResponse:
    ollama = ComponentHealth(ok=False, detail="unavailable")
    qdrant = ComponentHealth(ok=False, detail="unavailable")
    index = ComponentHealth(ok=False, detail="not indexed")
    try:
        models = await rag.ollama.models()
        missing = {rag.settings.embedding_model, rag.settings.generation_model} - models
        ollama = ComponentHealth(
            ok=not missing,
            detail="models ready"
            if not missing
            else f"missing models: {', '.join(sorted(missing))}",
        )
    except (OllamaError, RuntimeError) as exc:
        ollama.detail = type(exc).__name__
    try:
        metadata = await rag.store.metadata()
        qdrant = ComponentHealth(ok=True, detail="connected")
        index = ComponentHealth(
            ok=metadata is not None, detail="ready" if metadata else "not indexed"
        )
    except Exception as exc:
        qdrant.detail = type(exc).__name__
    ready = ollama.ok and qdrant.ok and index.ok
    return HealthResponse(
        status="ready" if ready else "degraded",
        backend=ComponentHealth(ok=True, detail="running"),
        ollama=ollama,
        qdrant=qdrant,
        index=index,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/api/config")
async def config(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return settings.public_dict()


@app.get("/api/documents", response_model=DocumentStatus)
async def documents(rag: RagService = Depends(service)) -> DocumentStatus:
    try:
        metadata = await rag.store.metadata()
        return DocumentStatus(indexed=metadata is not None, metadata=metadata)
    except Exception as exc:
        raise AppError(
            ErrorCode.QDRANT_UNAVAILABLE,
            "The vector store is unavailable.",
            status_code=503,
            retryable=True,
        ) from exc


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, rag: RagService = Depends(service)) -> IngestResponse:
    logger.info("ingestion requested force=%s", body.force)
    return await rag.ingest(body.force, body.chunk_size, body.chunk_overlap)


@app.post("/api/query", response_model=QueryResponse)
async def query(
    body: QueryRequest, request: Request, rag: RagService = Depends(service)
) -> QueryResponse:
    try:
        return await rag.query(
            body.question,
            body.top_k,
            body.score_threshold,
            request_id(request),
        )
    except AppError:
        metrics.increment("rag_queries_failed_total")
        raise


@app.post("/api/evaluation/retrieve")
async def evaluation_retrieve(
    body: EvaluationRetrievalRequest, rag: RagService = Depends(service)
) -> dict[str, object]:
    return await rag.evaluate_retrieval(body.question, body.rerank, body.top_k)


@app.post("/api/query/stream")
async def query_stream(
    body: QueryRequest, request: Request, rag: RagService = Depends(service)
) -> StreamingResponse:
    if not rag.settings.streaming_enabled:
        raise AppError(
            ErrorCode.INVALID_REQUEST,
            "Streaming is disabled; use /api/query.",
            status_code=409,
        )

    async def events():
        async for event in rag.stream_query(
            body.question,
            body.top_k,
            body.score_threshold,
            request_id(request),
        ):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
