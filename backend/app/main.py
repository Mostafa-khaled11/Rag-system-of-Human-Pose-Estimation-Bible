from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.ollama import OllamaClient
from app.core.vector_store import VectorStore
from app.retrieval.service import RagService
from app.schemas.models import (
    ComponentHealth,
    DocumentStatus,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(
    level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ollama = OllamaClient(settings.ollama_base_url, settings.ollama_timeout_seconds)
    store = VectorStore(
        settings.qdrant_url, settings.collection_name, settings.qdrant_timeout_seconds
    )
    app.state.rag = RagService(settings, ollama, store)
    yield
    await ollama.close()
    await store.close()


app = FastAPI(title="Human Pose Estimation RAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def service(request: Request) -> RagService:
    return request.app.state.rag


@app.get("/health", response_model=HealthResponse)
async def health(rag: RagService = Depends(service)) -> HealthResponse:
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
    except (httpx.HTTPError, RuntimeError) as exc:
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


@app.get("/api/config")
async def config(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return settings.public_dict()


@app.get("/api/documents", response_model=DocumentStatus)
async def documents(rag: RagService = Depends(service)) -> DocumentStatus:
    try:
        metadata = await rag.store.metadata()
        return DocumentStatus(indexed=metadata is not None, metadata=metadata)
    except Exception as exc:
        raise HTTPException(503, "Qdrant is unavailable") from exc


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, rag: RagService = Depends(service)) -> IngestResponse:
    logger.info("ingestion requested force=%s", body.force)
    try:
        return await rag.ingest(body.force, body.chunk_size, body.chunk_overlap)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ingestion failed")
        raise HTTPException(503, f"Ingestion failed: {type(exc).__name__}") from exc


@app.post("/api/query", response_model=QueryResponse)
async def query(body: QueryRequest, rag: RagService = Depends(service)) -> QueryResponse:
    try:
        return await rag.query(body.question, body.top_k, body.score_threshold)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("query failed")
        raise HTTPException(503, f"Query failed: {type(exc).__name__}") from exc
