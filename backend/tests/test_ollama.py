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


@pytest.mark.asyncio
async def test_retry_is_bounded_for_transient_failures() -> None:
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": "busy"})

    client = OllamaClient("http://test", 1, retry_count=1, retry_backoff=0)
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )
    with pytest.raises(OllamaError):
        await client.embed("model", ["text"])
    assert calls == 2
    await client.close()


@pytest.mark.asyncio
async def test_model_not_found_is_not_retried() -> None:
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"error": "missing"})

    client = OllamaClient("http://test", 1, retry_count=3, retry_backoff=0)
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )
    with pytest.raises(OllamaError) as caught:
        await client.embed("missing", ["text"])
    assert caught.value.status_code == 404
    assert calls == 1
    await client.close()
