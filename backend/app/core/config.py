from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    app_name: str = "Human Pose Estimation RAG"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    book_path: Path = Path("/data/source/HPE-Bible.pdf")
    max_source_bytes: int = 100_000_000
    max_question_chars: int = 2_000
    ollama_base_url: str = "http://ollama:11434"
    generation_model: str = "qwen2.5:1.5b"
    embedding_model: str = "nomic-embed-text:latest"
    ollama_timeout_seconds: float = 120.0
    ollama_temperature: float = 0.1
    qdrant_url: str = "http://qdrant:6333"
    qdrant_timeout_seconds: float = 15.0
    collection_name: str = "human_pose_estimation"
    chunk_strategy: Literal["recursive"] = "recursive"
    chunk_size: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    min_chunk_size: int = Field(default=200, ge=1, le=4000)
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    retrieval_top_k: int = Field(default=5, ge=1, le=25)
    retrieval_score_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=12000, ge=1000, le=50000)

    @model_validator(mode="after")
    def validate_chunking(self) -> Settings:
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("MIN_CHUNK_SIZE cannot exceed CHUNK_SIZE")
        return self

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def public_dict(self) -> dict[str, object]:
        values = self.model_dump(exclude={"book_path", "cors_origins"})
        values["source_filename"] = self.book_path.name
        return values


@lru_cache
def get_settings() -> Settings:
    return Settings()
