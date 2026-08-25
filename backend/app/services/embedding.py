from __future__ import annotations

import httpx

from app.core.errors import AppError, ErrorCode
from app.core.ollama import OllamaClient, OllamaError


class EmbeddingService:
    def __init__(self, client: OllamaClient, model: str) -> None:
        self.client = client
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self.client.embed(self.model, texts)
        except httpx.TimeoutException as exc:
            raise AppError(
                ErrorCode.QUERY_TIMEOUT,
                "Embedding request timed out.",
                status_code=504,
                retryable=True,
            ) from exc
        except OllamaError as exc:
            lowered = str(exc).lower()
            if "timed out" in lowered:
                raise AppError(
                    ErrorCode.QUERY_TIMEOUT,
                    "Embedding request timed out.",
                    status_code=504,
                    retryable=True,
                ) from exc
            if exc.status_code == 404:
                code = ErrorCode.MODEL_NOT_FOUND
            elif any(word in lowered for word in ("offline", "unavailable", "request failed")):
                code = ErrorCode.OLLAMA_UNAVAILABLE
            else:
                code = ErrorCode.EMBEDDING_FAILED
            message = (
                "The embedding model is unavailable."
                if code == ErrorCode.MODEL_NOT_FOUND
                else (
                    "The local model service is unavailable."
                    if code == ErrorCode.OLLAMA_UNAVAILABLE
                    else "Embedding failed."
                )
            )
            raise AppError(
                code,
                message,
                status_code=503,
                retryable=code not in {ErrorCode.MODEL_NOT_FOUND},
            ) from exc
