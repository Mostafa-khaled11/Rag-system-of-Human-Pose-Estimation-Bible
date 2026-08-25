from pathlib import Path


def test_compose_preserves_named_model_and_index_volumes() -> None:
    compose = (Path(__file__).parents[2] / "docker-compose.yml").read_text()
    assert "ollama_data:/root/.ollama" in compose
    assert "qdrant_data:/qdrant/storage" in compose
    assert "ollama_data:" in compose
    assert "qdrant_data:" in compose
    assert "down -v" not in compose
