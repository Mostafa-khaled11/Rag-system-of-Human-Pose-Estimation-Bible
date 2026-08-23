# Human Pose Estimation Book RAG

A private, local Retrieval-Augmented Generation application grounded only in the supplied Human Pose Estimation PDF. PDF text is split by page into overlapping character chunks, embedded by Ollama (`nomic-embed-text:latest`), persisted in Qdrant, retrieved by cosine similarity, and answered by Ollama (`qwen2.5:1.5b`). The React UI shows service/index state, answers, excerpts, and page citations.

No hosted inference, telemetry, analytics, CDN, or cloud vector service is used. After images and models are downloaded, runtime can operate offline.

## Requirements

- Docker Desktop with Compose v2
- Roughly 6–8 GB free RAM for comfortable CPU use and several GB of disk for images/models/vectors
- The unencrypted, text-based PDF. The supplied `HPE-Bible.pdf` was inspected as a 495-page, 9.0 MB, unencrypted PDF with extractable text.

CPU inference works but can be slow. GPU setup is optional and platform-specific; add the Ollama GPU device/reservation supported by your Docker host without changing application services or models.

## Start

1. Copy `.env.example` to `.env`.
2. Set `BOOK_HOST_PATH` to the absolute host path of the PDF. Docker Desktop must have access to that drive. On Windows, forward slashes are simplest, for example `D:/discussion/HPE-Bible.pdf`.
3. Start the stack:

```sh
docker compose up --build -d
docker compose logs -f ollama-init
```

`ollama-init` pulls both exact required models once into the persistent `ollama_data` volume and prints `ollama list`. Open <http://localhost:3000> after services become healthy. API docs are at <http://localhost:8000/docs>.

Index from the UI or API:

```sh
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" -d '{"force":false}'

curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How are human poses represented?","top_k":5}'
```

Stop without deleting data using `docker compose down`. Restart with `docker compose up -d`; named volumes retain models and vectors.

## Architecture and safety

The backend reads only the configured read-only mount; the API never accepts filesystem paths or uploads. Text is normalized conservatively, page numbers are preserved, headings are detected heuristically, and IDs derive from document SHA-256 + chunk fingerprint + page + chunk index. Repeated ingestion is idempotent. A changed fingerprint is never silently treated as compatible. Indexing builds a temporary collection before activating it so an interrupted embedding run does not overwrite the active index.

Retrieved book content is marked as untrusted reference material in the prompt. The model receives stable context IDs (`C1`, etc.); the backend maps only valid IDs to real pages and removes invalid citation markers. Full book passages are never logged.

## Configuration

All settings are listed in `.env.example` and returned without paths by `GET /api/config`. Defaults use 1,200-character chunks, 200-character overlap, minimum 200 characters, embedding batches of 16, top 5 retrieval, score threshold 0, 12,000 context characters, and temperature 0.1. Chunk settings changed in the UI require re-indexing. Host development should use `OLLAMA_BASE_URL=http://localhost:11434`, `QDRANT_URL=http://localhost:6333`, and an absolute host `BOOK_PATH`.

Endpoints: `GET /health`, `GET /api/config`, `GET /api/documents`, `POST /api/ingest`, and `POST /api/query`.

## Development checks

```sh
cd backend
python -m pip install -e ".[dev]"
ruff check .
pytest

cd ../frontend
npm install
npm run lint
npm test
npm run build

docker compose config
docker compose build
```

The backend tests mock local services and cover validation, boundaries/overlap, page/section metadata, deterministic IDs, malformed embeddings, citation serialization, empty retrieval, and dimension mismatch behavior.

## Troubleshooting and limitations

- **Missing/encrypted/corrupt/scanned PDF:** ingestion returns a clear 422. Scanned books require a separately prepared local OCR copy; OCR is not silently attempted.
- **Sparse pages:** page numbers with fewer than 50 extracted characters are reported. Diagrams, equation layout, multi-column order, and visual details may not be represented correctly by PDF text extraction.
- **Ollama/model unavailable:** inspect `docker compose logs ollama ollama-init`; downloads may be slow and require internet only initially.
- **Qdrant unavailable/dimension mismatch:** inspect Qdrant logs. A model/dimension change requires explicit force re-indexing.
- **Weak retrieval:** adjust top-k or score threshold; an empty result returns `insufficient_context` without calling the generation model.
- **Port/CORS problems:** change host port mappings and add only local origins to `CORS_ORIGINS`.
- **Resource pressure/context:** reduce embedding batch size or context characters. CPU generation can take minutes on modest hardware.
- **Interrupted ingestion:** the active index remains until the staged collection is complete; a later ingestion cleans its deterministic staging target.

### Destructive reset

`docker compose down -v` permanently deletes downloaded models and all indexed vectors. It does not delete the host PDF, but use it only when a full reset is intended.

Known limitations include heuristic headings, character—not token—chunk sizes, no OCR, no diagram understanding, and single-document configuration. A useful next improvement is local OCR/figure caption extraction behind an explicit opt-in, followed by reranking evaluation against a small book-specific question set.

