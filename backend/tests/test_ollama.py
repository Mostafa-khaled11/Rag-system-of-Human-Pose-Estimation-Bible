import httpx
import pytest

from app.core.ollama import OllamaClient, OllamaError


@pytest.mark.asyncio
async def test_malformed_embedding_response() -> None:
    client = OllamaClient("http://test", 1)
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"embeddings": [[1.0], [1.0, 2.0]]})
        ),
        base_url="http://test",
    )
    with pytest.raises(OllamaError, match="dimensions differ"):
        await client.embed("model", ["a", "b"])
    await client.close()
