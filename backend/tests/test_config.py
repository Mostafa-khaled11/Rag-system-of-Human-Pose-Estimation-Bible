import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.embedding_model == "nomic-embed-text:latest"
    assert settings.generation_model == "qwen2.5:1.5b"


def test_overlap_must_be_smaller() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=300, chunk_overlap=300)
