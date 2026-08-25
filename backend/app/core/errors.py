from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    OLLAMA_UNAVAILABLE = "OLLAMA_UNAVAILABLE"
    QDRANT_UNAVAILABLE = "QDRANT_UNAVAILABLE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    GENERATION_FAILED = "GENERATION_FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    QUERY_TIMEOUT = "QUERY_TIMEOUT"
    INGESTION_FAILED = "INGESTION_FAILED"


class AppError(RuntimeError):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        status_code: int = 503,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details


def invalid_request(message: str, details: dict[str, Any] | None = None) -> AppError:
    return AppError(ErrorCode.INVALID_REQUEST, message, status_code=422, details=details)
