from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from app.core.errors import AppError, ErrorCode
from app.core.ollama import OllamaClient, OllamaError
from app.core.prompts import GroundedAnswerPrompt


class GenerationService:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        temperature: float,
        prompt: GroundedAnswerPrompt | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.prompt = prompt or GroundedAnswerPrompt()

    def build_prompt(self, question: str, context: str) -> str:
        return self.prompt.render(question, context)

    async def generate(self, question: str, context: str) -> str:
        try:
            return await self.client.generate(
                self.model, self.build_prompt(question, context), self.temperature
            )
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise AppError(
                ErrorCode.QUERY_TIMEOUT,
                "Answer generation timed out.",
                status_code=504,
                retryable=True,
            ) from exc
        except OllamaError as exc:
            message = str(exc).lower()
            if "timed out" in message:
                raise AppError(
                    ErrorCode.QUERY_TIMEOUT,
                    "Answer generation timed out.",
                    status_code=504,
                    retryable=True,
                ) from exc
            if exc.status_code == 404:
                code = ErrorCode.MODEL_NOT_FOUND
            elif any(word in message for word in ("offline", "unavailable", "request failed")):
                code = ErrorCode.OLLAMA_UNAVAILABLE
            else:
                code = ErrorCode.GENERATION_FAILED
            error_message = (
                "The generation model is unavailable."
                if code == ErrorCode.MODEL_NOT_FOUND
                else (
                    "The local model service is unavailable."
                    if code == ErrorCode.OLLAMA_UNAVAILABLE
                    else "Answer generation failed."
                )
            )
            raise AppError(
                code,
                error_message,
                status_code=503,
                retryable=code != ErrorCode.MODEL_NOT_FOUND,
            ) from exc

    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        try:
            async for chunk in self.client.generate_stream(
                self.model, self.build_prompt(question, context), self.temperature
            ):
                yield chunk
        except OllamaError as exc:
            message = str(exc).lower()
            if "timed out" in message:
                code = ErrorCode.QUERY_TIMEOUT
            elif exc.status_code == 404:
                code = ErrorCode.MODEL_NOT_FOUND
            elif any(word in message for word in ("offline", "unavailable", "request failed")):
                code = ErrorCode.OLLAMA_UNAVAILABLE
            else:
                code = ErrorCode.GENERATION_FAILED
            raise AppError(
                code,
                "Answer generation timed out."
                if code == ErrorCode.QUERY_TIMEOUT
                else (
                    "The local model service is unavailable."
                    if code == ErrorCode.OLLAMA_UNAVAILABLE
                    else "Answer generation failed."
                ),
                status_code=504 if code == ErrorCode.QUERY_TIMEOUT else 503,
                retryable=code != ErrorCode.MODEL_NOT_FOUND,
            ) from exc
