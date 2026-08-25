from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx


class OllamaError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        connect_timeout: float = 5.0,
        retry_count: int = 2,
        retry_backoff: float = 0.5,
    ) -> None:
        self.retry_count = retry_count
        self.retry_backoff = retry_backoff
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def _post(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                response = await self.client.post(path, json=payload)
                if response.status_code in {400, 404}:
                    raise OllamaError(
                        f"Ollama rejected the request ({response.status_code})",
                        status_code=response.status_code,
                    )
                response.raise_for_status()
                return response.json()
            except OllamaError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt < self.retry_count:
                    await asyncio.sleep(self.retry_backoff * (2**attempt))
        raise OllamaError(f"Ollama request failed: {last}") from last

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = await self._post("/api/embed", {"model": model, "input": texts})
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError(
                "Malformed embedding response: vector count does not match input count"
            )
        dimensions = {len(vector) for vector in vectors if isinstance(vector, list)}
        if (
            len(dimensions) != 1
            or not dimensions
            or 0 in dimensions
            or len(vectors) != sum(isinstance(v, list) for v in vectors)
        ):
            raise OllamaError(
                "Malformed embedding response: vectors are empty or dimensions differ"
            )
        if any(not isinstance(value, int | float) for vector in vectors for value in vector):
            raise OllamaError("Malformed embedding response: vectors contain non-numeric values")
        return vectors

    async def generate(self, model: str, prompt: str, temperature: float) -> str:
        data = await self._post(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        answer = data.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise OllamaError("Ollama returned an empty generation response")
        return answer.strip()

    async def generate_stream(
        self, model: str, prompt: str, temperature: float
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            async with self.client.stream("POST", "/api/generate", json=payload) as response:
                if response.status_code in {400, 404}:
                    raise OllamaError(
                        f"Ollama rejected the request ({response.status_code})",
                        status_code=response.status_code,
                    )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError as exc:
                        raise OllamaError("Malformed Ollama streaming response") from exc
                    if event.get("error"):
                        raise OllamaError(str(event["error"]))
                    chunk = event.get("response", "")
                    if chunk:
                        yield str(chunk)
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama generation timed out") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama streaming request failed: {exc}") from exc

    async def models(self) -> set[str]:
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            return {item.get("name", "") for item in response.json().get("models", [])}
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama health check failed: {exc}") from exc
