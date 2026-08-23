from __future__ import annotations

import asyncio

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def _post(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for delay in (0.0, 0.5, 1.5):
            if delay:
                await asyncio.sleep(delay)
            try:
                response = await self.client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
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

    async def models(self) -> set[str]:
        response = await self.client.get("/api/tags")
        response.raise_for_status()
        return {item.get("name", "") for item in response.json().get("models", [])}
