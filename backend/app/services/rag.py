from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, invalid_request
from app.core.metrics import metrics
from app.core.ollama import OllamaClient
from app.core.prompts import GroundedAnswerPrompt
from app.core.vector_store import SearchHit, VectorStore
from app.schemas.models import Citation, QueryResponse, RetrievedChunk, Timing
from app.services.embedding import EmbeddingService
from app.services.generation import GenerationService
from app.services.ingestion import IngestionService
from app.services.reranking import FallbackRerankingService, RerankingService
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


@dataclass
class PreparedQuery:
    question: str
    request_id: str
    started: float
    context: str
    selected: list[SearchHit]
    candidate_count: int
    timing: Timing
    reranking_applied: bool


class RagService:
    def __init__(
        self,
        settings: Settings,
        ollama: OllamaClient,
        store: VectorStore,
        reranker: FallbackRerankingService | None = None,
    ) -> None:
        self.settings = settings
        self.ollama = ollama
        self.store = store
        self.embedding = EmbeddingService(ollama, settings.embedding_model)
        self.retrieval = RetrievalService(store)
        if reranker is None:
            try:
                primary = RerankingService(
                    settings.reranking_enabled,
                    settings.reranker_model,
                    settings.reranker_vector_weight,
                )
            except Exception:
                logger.exception("reranker initialization failed; vector fallback enabled")
                primary = None
            reranker = FallbackRerankingService(primary)
        self.reranker = reranker
        self.generation = GenerationService(
            ollama,
            settings.generation_model,
            settings.ollama_temperature,
        )
        self.ingestion = IngestionService(settings, self.embedding, store)

    async def ingest(self, force: bool, chunk_size: int | None, chunk_overlap: int | None):
        return await self.ingestion.ingest(force, chunk_size, chunk_overlap)

    async def prepare(
        self,
        question: str,
        top_k: int | None,
        threshold: float | None,
        request_id: str | None = None,
    ) -> PreparedQuery:
        started = time.perf_counter()
        question = question.strip()
        if len(question) < 2:
            raise invalid_request("Question must contain at least 2 non-whitespace characters.")
        if len(question) > self.settings.max_question_chars:
            raise invalid_request(
                f"Question exceeds {self.settings.max_question_chars} characters.",
                {"max_question_chars": self.settings.max_question_chars},
            )
        request_id = request_id or str(uuid.uuid4())
        embed_started = time.perf_counter()
        vector = (await self.embedding.embed([question]))[0]
        embedded = time.perf_counter()
        final_k = top_k or self.settings.retrieval_top_k
        candidate_k = (
            max(final_k, self.settings.retrieval_candidate_count)
            if self.settings.reranking_enabled
            else final_k
        )
        hits = await self.retrieval.retrieve(
            vector,
            candidate_k,
            self.settings.retrieval_score_threshold if threshold is None else threshold,
        )
        retrieved = time.perf_counter()
        rerank_result = self.reranker.rerank(question, hits, final_k)
        reranked = time.perf_counter()
        context, selected = self._pack_context(rerank_result.hits)
        return PreparedQuery(
            question=question,
            request_id=request_id,
            started=started,
            context=context,
            selected=selected,
            candidate_count=len(hits),
            timing=Timing(
                embedding_ms=(embedded - embed_started) * 1000,
                retrieval_ms=(retrieved - embedded) * 1000,
                reranking_ms=(reranked - retrieved) * 1000,
            ),
            reranking_applied=rerank_result.applied,
        )

    def has_sufficient_evidence(self, prepared: PreparedQuery) -> bool:
        lexical = RerankingService(
            True, "lexical-v1", self.settings.reranker_vector_weight
        )
        return (
            len(prepared.selected) >= self.settings.minimum_evidence_passages
            and bool(prepared.context)
            and max((hit.score for hit in prepared.selected), default=-1.0)
            >= self.settings.minimum_evidence_score
            and max(
                (lexical.lexical_score(prepared.question, hit) for hit in prepared.selected),
                default=0.0,
            )
            >= self.settings.minimum_lexical_overlap
        )

    async def query(
        self,
        question: str,
        top_k: int | None,
        threshold: float | None,
        request_id: str | None = None,
    ) -> QueryResponse:
        metrics.increment("rag_queries_total")
        prepared = await self.prepare(question, top_k, threshold, request_id)
        if not self.has_sufficient_evidence(prepared):
            response = self._insufficient(prepared)
            self._record(response, prepared, success=True)
            return response
        generation_started = time.perf_counter()
        raw_answer = await self.generation.generate(prepared.question, prepared.context)
        response = self._finalize(prepared, raw_answer, generation_started)
        self._record(response, prepared, success=True)
        return response

    async def evaluate_retrieval(
        self, question: str, rerank: bool, top_k: int | None
    ) -> dict[str, Any]:
        question = question.strip()
        if len(question) < 2 or len(question) > self.settings.max_question_chars:
            raise invalid_request("The evaluation question length is invalid.")
        vector = (await self.embedding.embed([question]))[0]
        final_k = top_k or self.settings.retrieval_top_k
        candidate_k = max(final_k, self.settings.retrieval_candidate_count) if rerank else final_k
        hits = await self.retrieval.retrieve(
            vector, candidate_k, self.settings.retrieval_score_threshold
        )
        evaluation_reranker = RerankingService(
            True, self.settings.reranker_model, self.settings.reranker_vector_weight
        )
        result = evaluation_reranker.rerank(question, hits, final_k) if rerank else None
        selected = result.hits if result else hits[:final_k]
        return {
            "pages": [int(hit.payload["page"]) for hit in selected],
            "scores": [hit.score for hit in selected],
            "chunk_ids": [hit.id for hit in selected],
            "candidate_count": len(hits),
            "reranking_applied": bool(result and result.applied),
        }

    async def stream_query(
        self,
        question: str,
        top_k: int | None,
        threshold: float | None,
        request_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        request_id = request_id or str(uuid.uuid4())
        metrics.increment("rag_queries_total")
        yield {"type": "status", "phase": "retrieving", "request_id": request_id}
        try:
            prepared = await self.prepare(question, top_k, threshold, request_id)
            yield {
                "type": "retrieval",
                "phase": "generating" if self.has_sufficient_evidence(prepared) else "completed",
                "candidate_count": prepared.candidate_count,
                "passages": [
                    chunk.model_dump(mode="json") for chunk in self._chunks(prepared.selected)
                ],
                "timing": prepared.timing.model_dump(mode="json"),
                "request_id": request_id,
            }
            if not self.has_sufficient_evidence(prepared):
                response = self._insufficient(prepared)
                self._record(response, prepared, success=True)
                yield {"type": "final", "data": response.model_dump(mode="json")}
                yield {"type": "done", "request_id": request_id}
                return
            generation_started = time.perf_counter()
            chunks: list[str] = []
            async for chunk in self.generation.stream(prepared.question, prepared.context):
                chunks.append(chunk)
                yield {"type": "token", "text": chunk, "request_id": request_id}
            response = self._finalize(prepared, "".join(chunks), generation_started)
            self._record(response, prepared, success=True)
            yield {"type": "final", "data": response.model_dump(mode="json")}
            yield {"type": "done", "request_id": request_id}
        except AppError as exc:
            metrics.increment("rag_queries_failed_total")
            logger.warning(
                "streaming query failed",
                extra={"request_id": request_id, "error_code": exc.code},
            )
            yield {
                "type": "error",
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "request_id": request_id,
                    "details": exc.details,
                },
            }
        except Exception:
            metrics.increment("rag_queries_failed_total")
            logger.exception("unexpected streaming query failure", extra={"request_id": request_id})
            yield {
                "type": "error",
                "error": {
                    "code": ErrorCode.GENERATION_FAILED,
                    "message": "The query failed unexpectedly.",
                    "retryable": True,
                    "request_id": request_id,
                    "details": None,
                },
            }

    def _finalize(
        self, prepared: PreparedQuery, raw_answer: str, generation_started: float
    ) -> QueryResponse:
        if GroundedAnswerPrompt().insufficient_message.lower() in raw_answer.lower():
            return self._insufficient(prepared, generation_started)
        allowed = {f"C{i}" for i in range(1, len(prepared.selected) + 1)}
        used = {item for item in re.findall(r"\[(C\d+)\]", raw_answer) if item in allowed}
        answer = re.sub(
            r"\[(C\d+)\]",
            lambda match: (
                f"[p. {prepared.selected[int(match.group(1)[1:]) - 1].payload['page']}]"
                if match.group(1) in allowed
                else ""
            ),
            raw_answer,
        ).strip()
        cited_indices = sorted(int(item[1:]) - 1 for item in used)
        if not cited_indices:
            cited_indices = list(range(len(prepared.selected)))
            pages = list(dict.fromkeys(int(hit.payload["page"]) for hit in prepared.selected))
            answer = f"{answer}\n\nSupporting pages: {', '.join(f'[p. {page}]' for page in pages)}"
        citations = [self._citation(prepared.selected[index]) for index in cited_indices]
        finished = time.perf_counter()
        prepared.timing.generation_ms = (finished - generation_started) * 1000
        prepared.timing.total_ms = (finished - prepared.started) * 1000
        return QueryResponse(
            status="grounded",
            answer=answer,
            citations=citations,
            retrieved_chunks=self._chunks(prepared.selected),
            generation_model=self.settings.generation_model,
            embedding_model=self.settings.embedding_model,
            timing=prepared.timing,
            request_id=prepared.request_id,
            answerable=True,
            confidence=self._confidence(prepared.selected),
            insufficient_context=False,
            reranking_applied=prepared.reranking_applied,
            reranker_model=self.settings.reranker_model if prepared.reranking_applied else None,
        )

    def _pack_context(self, hits: list[SearchHit]) -> tuple[str, list[SearchHit]]:
        blocks: list[str] = []
        selected: list[SearchHit] = []
        used = 0
        for hit in hits:
            text = str(hit.payload["chunk_text"])
            label = hit.payload.get("chapter") or "unknown"
            block = f"[C{len(selected) + 1}] Page {hit.payload['page']}; section: {label}\n{text}"
            if used + len(block) > self.settings.max_context_chars:
                continue
            blocks.append(block)
            selected.append(hit)
            used += len(block)
        return "\n\n".join(blocks), selected

    @staticmethod
    def _citation(hit: SearchHit) -> Citation:
        payload = hit.payload
        return Citation(
            page=int(payload["page"]),
            chapter=payload.get("chapter"),
            chunk_id=hit.id,
            score=hit.score,
            excerpt=str(payload["chunk_text"])[:400],
        )

    def _chunks(self, hits: list[SearchHit]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                **self._citation(hit).model_dump(),
                source_filename=str(hit.payload.get("source_filename", "book.pdf")),
            )
            for hit in hits
        ]

    @staticmethod
    def _confidence(hits: list[SearchHit]) -> float:
        return round(max(0.0, min(1.0, max((hit.score for hit in hits), default=0.0))), 3)

    def _insufficient(
        self, prepared: PreparedQuery, generation_started: float | None = None
    ) -> QueryResponse:
        finished = time.perf_counter()
        if generation_started is not None:
            prepared.timing.generation_ms = (finished - generation_started) * 1000
        prepared.timing.total_ms = (finished - prepared.started) * 1000
        return QueryResponse(
            status="insufficient_context",
            answer=GroundedAnswerPrompt().insufficient_message,
            citations=[],
            retrieved_chunks=self._chunks(prepared.selected),
            generation_model=self.settings.generation_model,
            embedding_model=self.settings.embedding_model,
            timing=prepared.timing,
            request_id=prepared.request_id,
            answerable=False,
            confidence=self._confidence(prepared.selected),
            insufficient_context=True,
            reranking_applied=prepared.reranking_applied,
            reranker_model=self.settings.reranker_model if prepared.reranking_applied else None,
        )

    def _record(
        self, response: QueryResponse, prepared: PreparedQuery, *, success: bool
    ) -> None:
        metrics.increment("rag_queries_success_total" if success else "rag_queries_failed_total")
        if response.insufficient_context:
            metrics.increment("rag_insufficient_context_total")
        for name, value in (
            ("rag_query_latency_seconds", response.timing.total_ms),
            ("rag_embedding_latency_seconds", response.timing.embedding_ms),
            ("rag_retrieval_latency_seconds", response.timing.retrieval_ms),
            ("rag_reranking_latency_seconds", response.timing.reranking_ms),
            ("rag_generation_latency_seconds", response.timing.generation_ms),
        ):
            metrics.observe_ms(name, value)
        logger.info(
            "query completed",
            extra={
                "request_id": prepared.request_id,
                "question_length": len(prepared.question),
                "embedding_ms": response.timing.embedding_ms,
                "retrieval_ms": response.timing.retrieval_ms,
                "reranking_ms": response.timing.reranking_ms,
                "generation_ms": response.timing.generation_ms,
                "total_ms": response.timing.total_ms,
                "candidate_count": prepared.candidate_count,
                "passage_count": len(prepared.selected),
                "pages": [int(hit.payload["page"]) for hit in prepared.selected],
                "scores": [round(hit.score, 4) for hit in prepared.selected],
            },
        )
