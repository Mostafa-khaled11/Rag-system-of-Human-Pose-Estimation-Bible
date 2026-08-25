"""Backward-compatible import for the refactored RAG orchestration service."""

from app.services.rag import RagService

__all__ = ["RagService"]
