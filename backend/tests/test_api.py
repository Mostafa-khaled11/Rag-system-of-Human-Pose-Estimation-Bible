from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.schemas.models import QueryResponse, Timing


class FakeOllama:
    fail = False

    async def models(self):
        if self.fail:
            raise RuntimeError("offline")
        return {"qwen2.5:1.5b", "nomic-embed-text:latest"}


class FakeStore:
    fail = False

    async def metadata(self):
        if self.fail:
            raise RuntimeError("offline")
        return {"chunk_count": 1}


class FakeRag:
    def __init__(self):
        self.settings = Settings(_env_file=None)
        self.ollama = FakeOllama()
        self.store = FakeStore()

    async def query(self, _question, _top_k, _threshold, request_id):
        return QueryResponse(
            status="grounded",
            answer="answer [p. 7]",
            citations=[],
            retrieved_chunks=[],
            generation_model=self.settings.generation_model,
            embedding_model=self.settings.embedding_model,
            timing=Timing(total_ms=1),
            request_id=request_id,
        )

    async def stream_query(self, _question, _top_k, _threshold, request_id):
        yield {"type": "token", "text": "answer", "request_id": request_id}
        response = await self.query("", None, None, request_id)
        yield {"type": "final", "data": response.model_dump(mode="json")}
        yield {"type": "done", "request_id": request_id}

    async def evaluate_retrieval(self, _question, rerank, _top_k):
        return {
            "pages": [7],
            "scores": [0.9],
            "chunk_ids": ["c1"],
            "candidate_count": 20 if rerank else 5,
            "reranking_applied": rerank,
        }


@contextmanager
def client(rag: FakeRag | None = None):
    with TestClient(app) as context:
        context.app.state.rag = rag or FakeRag()
        yield context


def test_health_is_lightweight_and_ready_checks_dependencies() -> None:
    with client() as test_client:
        assert test_client.get("/health").json()["status"] == "ok"
        ready = test_client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"


def test_config_documents_query_and_metrics_endpoints() -> None:
    with client() as test_client:
        assert test_client.get("/api/config").status_code == 200
        assert test_client.get("/api/documents").json()["indexed"]
        response = test_client.post("/api/query", json={"question": "valid question"})
        assert response.status_code == 200
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
        evaluation = test_client.post(
            "/api/evaluation/retrieve",
            json={"question": "valid question", "rerank": False},
        )
        assert evaluation.json()["pages"] == [7]
        assert "rag_" in test_client.get("/metrics").text


def test_streaming_endpoint_is_ndjson_and_disables_proxy_buffering() -> None:
    with client() as test_client:
        response = test_client.post("/api/query/stream", json={"question": "valid question"})
        assert response.status_code == 200
        assert response.headers["X-Accel-Buffering"] == "no"
        assert '"type": "token"' in response.text
        assert '"type": "final"' in response.text


def test_validation_errors_are_structured() -> None:
    with client() as test_client:
        for body in ({"question": ""}, {}, {"question": "x" * 2001}):
            response = test_client.post("/api/query", json=body)
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "INVALID_REQUEST"
            assert response.json()["error"]["request_id"]
        malformed = test_client.post(
            "/api/query", content="{", headers={"Content-Type": "application/json"}
        )
        assert malformed.status_code == 422


def test_dependency_failures_are_visible_in_readiness_and_documents() -> None:
    rag = FakeRag()
    rag.ollama.fail = True
    rag.store.fail = True
    with client(rag) as test_client:
        ready = test_client.get("/ready").json()
        assert ready["status"] == "degraded"
        assert not ready["ollama"]["ok"]
        response = test_client.get("/api/documents")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "QDRANT_UNAVAILABLE"


def test_nginx_streaming_timeout_regression_configuration() -> None:
    nginx = (Path(__file__).parents[2] / "frontend" / "nginx.conf").read_text()
    assert "proxy_buffering off" in nginx
    assert "proxy_read_timeout 600s" in nginx
