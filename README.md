# Human Pose Estimation Book RAG

A private, local Retrieval-Augmented Generation (RAG) system for asking questions about a Human Pose Estimation reference book. The application retrieves relevant passages from the indexed PDF, asks a local language model to answer only from that context, and returns the answer with source excerpts and page citations.

Inference, embeddings, and vector storage run locally. No hosted model or cloud vector database is required after the container images and models have been downloaded.

## Project Overview

The project is designed around one primary Human Pose Estimation PDF. During ingestion, the backend extracts text page by page, divides it into overlapping character-based chunks, embeds those chunks, and stores them in Qdrant. At query time, the same embedding model converts the user's question to a vector, Qdrant finds the most relevant book chunks, and Qwen generates a grounded answer from the retrieved evidence.

The React interface exposes dependency and index status, indexing controls, progressively streamed grounded answers, retrieved excerpts, relevance scores, source page numbers, confidence, and query timing. Weak or irrelevant evidence is rejected before generation.

## Demo

[Watch the project demo](docs/images/demo.mp4)

## Architecture

- **React** provides the browser interface for indexing, configuration controls, questions, answers, and citations.
- **FastAPI** validates requests and delegates to focused embedding, retrieval, reranking, generation, and ingestion services.
- **Ollama** serves both local models inside Docker.
- **`nomic-embed-text:latest`** creates vectors for book chunks and user queries.
- **Qdrant** stores chunk vectors and metadata and returns a configurable candidate set.
- **`lexical-v1` reranker** combines query-token overlap with vector similarity without loading another large model. It is optional and falls back to vector order if initialization or execution fails.
- **`qwen2.5:1.5b`** generates an answer constrained by the retrieved passages.

The basic query flow is:

`User -> React -> FastAPI -> embedding -> Qdrant candidates -> reranker -> top context -> Qwen -> streamed grounded answer and citations`

## Architecture Diagram

```mermaid
flowchart LR
    subgraph Ingestion[Book ingestion]
        PDF[PDF Book] --> Extract[Page-aware Text Extraction]
        Extract --> Chunk[Character-based Chunking]
        Chunk --> ChunkEmbed[nomic-embed-text]
        ChunkEmbed --> Stage[Staged Qdrant Collection]
        Stage --> Activate[Activate Completed Index]
    end

    subgraph Query[Question answering]
        User[User] --> UI[React Frontend]
        UI --> API[FastAPI Backend]
        API --> QueryEmbed[nomic-embed-text]
        QueryEmbed --> VectorDB[Qdrant Active Index]
        Activate --> VectorDB
        VectorDB --> Candidates[Candidate Chunks]
        Candidates --> Rerank[CPU Reranker]
        Rerank --> Evidence{Enough Evidence?}
        Evidence -->|yes| LLM[qwen2.5:1.5b]
        Evidence -->|no| Insufficient[Insufficient Context]
        LLM --> Answer[Streaming Grounded Answer + Citations]
        Answer --> API
        API --> UI
    end

    subgraph Evaluation[Offline evaluation]
        Dataset[23 Book-Grounded Questions] --> Eval[Evaluation Runner]
        Eval --> Base[Vector Baseline]
        Eval --> New[Reranked Pipeline]
        Base --> Report[JSON + Markdown Report]
        New --> Report
    end
```

## Prerequisites

- Docker Desktop with Docker Compose v2. Docker Engine with the Compose plugin is also suitable on Linux.
- Git for cloning the repository.
- Internet access on the first start to download container images and the two Ollama models.
- An unencrypted, text-based PDF of the source book.
- Several gigabytes of free disk space for Docker images, Ollama models, and Qdrant data. Around 6-8 GB of available RAM is recommended for comfortable CPU-based local use; actual consumption depends on the host and workload.

Ollama runs as a service in `docker-compose.yml`; a host Ollama installation is not required for the standard Docker workflow. GPU acceleration is optional and is not configured by this repository. The stack works on Windows, macOS, or Linux where Docker Compose and Linux containers are supported. On Windows, Docker Desktop must be allowed to access any drive containing a PDF referenced by an absolute path.

## Models Used

| Model | Role |
| --- | --- |
| `qwen2.5:1.5b` | Generates answers from the retrieved context. |
| `nomic-embed-text:latest` | Embeds source chunks during indexing and questions during retrieval. |

These lightweight local models were selected because the current development environment has limited hardware capabilities and GPU acceleration is unavailable or insufficient for larger models. `qwen2.5:1.5b` is small enough for practical local generation, while `nomic-embed-text` is a lightweight fit for the local embedding pipeline.

This is primarily a hardware and development constraint, not a claim that these are the strongest models for production answer quality. Both can be replaced later when stronger GPU resources are available, although changing the embedding model or its vector dimension requires re-indexing.

## Setup

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/Mostafa-khaled11/Rag-system-of-Human-Pose-Estimation-Bible.git
   cd Rag-system-of-Human-Pose-Estimation-Bible
   ```

2. Create the local configuration file:

   PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   Bash:

   ```bash
   cp .env.example .env
   ```

3. Add the PDF as described in [Adding the Book](#adding-the-book). The default `.env.example` expects `data/source/HPE-Bible.pdf`.

4. Build and start the stack:

   ```bash
   docker compose up --build -d
   ```

5. Follow the one-time model download if needed:

   ```bash
   docker compose logs -f ollama-init
   ```

6. Open <http://localhost:3000>, wait for Ollama and Qdrant to report healthy, then select **Index book**.

Do not commit `.env`; it is intentionally ignored. The tracked `.env.example` contains all supported settings and safe local defaults.

## Required Ollama Models

The `ollama-init` service automatically downloads both required models during the first Docker startup and stores them in the persistent `ollama_data` volume. Later starts reuse the downloaded models.

No manual model pull is needed for the normal Docker setup. To refresh the models explicitly while the stack is running, use:

```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama pull nomic-embed-text:latest
```

If developing against a separately installed host Ollama instance instead, the equivalent host commands are:

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text:latest
```

Host-based backend development also requires overriding `OLLAMA_BASE_URL` and `QDRANT_URL` with host-accessible addresses.

## Adding the Book

Place a legally obtained, unencrypted, text-based PDF at:

```text
data/source/HPE-Bible.pdf
```

The default `BOOK_HOST_PATH=./data/source/HPE-Bible.pdf` bind-mounts that file read-only inside the backend container at `/data/source/HPE-Bible.pdf`. You may instead set `BOOK_HOST_PATH` in `.env` to another host path accessible to Docker Desktop.

`data/source/*` is excluded by `.gitignore` except for its `.gitkeep` file. Source books are intentionally not versioned; do not add or commit copyrighted PDFs.

## Indexing and Re-indexing

Indexing does not run automatically. Start it with the **Index book** button or call the API:

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"force":false}'
```

The ingestion pipeline:

1. Validates the source as a readable, unencrypted PDF with sufficient extractable text.
2. Extracts and normalizes text page by page while retaining page numbers.
3. Splits each page into chunks, using 1,200 characters and 200 characters of overlap by default.
4. Embeds the chunks in batches with `nomic-embed-text:latest`.
5. Builds a temporary Qdrant collection and records document and configuration metadata.
6. Activates the completed collection only after every chunk and the manifest have been stored successfully.

Chunk size is the approximate maximum number of characters in each passage. Overlap repeats trailing context in the following chunk to reduce information loss at boundaries. The implementation prefers paragraph or sentence boundaries when possible and merges small final fragments where they fit.

Repeated ingestion with the same PDF and chunk configuration is idempotent unless `force` is true. Changing chunk size or overlap requires re-indexing because stored chunks and embeddings must be rebuilt. The UI sends a forced re-index for an existing index. The active Qdrant alias is changed only after the staged index succeeds, so a failed run does not replace the usable index.

To force the current configuration to be rebuilt through the API:

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"force":true}'
```

## Running the Application

Start or resume all services:

```bash
docker compose up -d
```

Check container status:

```bash
docker compose ps
```

Stop containers without deleting models or vectors:

```bash
docker compose down
```

| Service | URL |
| --- | --- |
| React frontend | <http://localhost:3000> |
| FastAPI OpenAPI documentation | <http://localhost:8000/docs> |
| Backend liveness endpoint | <http://localhost:8000/health> |
| Dependency readiness endpoint | <http://localhost:8000/ready> |
| Prometheus-compatible metrics | <http://localhost:8000/metrics> |

The API also exposes `GET /api/config`, `GET /api/documents`, `POST /api/ingest`, the backward-compatible `POST /api/query`, and streaming `POST /api/query/stream`.

## Retrieval, Reranking, and Evidence Policy

When reranking is enabled, Qdrant retrieves `RETRIEVAL_CANDIDATE_COUNT` vector candidates (20 by default), and the local `lexical-v1` reranker combines normalized cosine similarity with query-token overlap before selecting `RETRIEVAL_TOP_K` passages (5 by default). It has no extra model download and is suitable for CPU-constrained development. Reranking defaults to off because the initial book evaluation showed unchanged Hit Rate but lower Recall and MRR; enable it only when calibration on the current index demonstrates a benefit.

Generation is skipped unless the packed context contains enough passages and passes both `MINIMUM_EVIDENCE_SCORE` and `MINIMUM_LEXICAL_OVERLAP`. The prompt permits only retrieved context IDs. Returned page citations are derived server-side from those IDs, so arbitrary model-supplied page numbers do not enter the citation list. Insufficient evidence returns `answerable=false`, `insufficient_context=true`, a confidence score, and the stable message:

> The retrieved passages do not contain enough information to answer this question reliably.

## Streaming Protocol

`POST /api/query/stream` returns newline-delimited JSON (`application/x-ndjson`). Events are typed as `status`, `retrieval`, `token`, `final`, `error`, and `done`. The final event contains the same structured response as `/api/query`, including citations, retrieved chunks, model names, confidence, reranker metadata, and stage timings. The React UI displays retrieval and generation states and renders source cards only after final metadata arrives.

Nginx disables proxy buffering, caching, request buffering, and gzip for this route and uses 600-second upstream timeouts. This prevents the former short-proxy-timeout failure mode during slow CPU inference while still allowing tokens to reach the browser immediately.

## Structured Errors and Reliability

API errors use this stable shape:

```json
{
  "error": {
    "code": "OLLAMA_UNAVAILABLE",
    "message": "A client-safe explanation.",
    "retryable": true,
    "request_id": "...",
    "details": null
  }
}
```

Error codes include `OLLAMA_UNAVAILABLE`, `QDRANT_UNAVAILABLE`, `MODEL_NOT_FOUND`, `EMBEDDING_FAILED`, `RETRIEVAL_FAILED`, `GENERATION_FAILED`, `INVALID_REQUEST`, `INSUFFICIENT_CONTEXT`, `INDEX_NOT_READY`, and `QUERY_TIMEOUT`. Ollama embedding and non-streaming generation calls use bounded exponential-backoff retries. Connect, generation/read, Qdrant, retry-count, and backoff settings are configurable. Validation and missing-model errors are not retried.

Every response includes `X-Request-ID`; structured JSON logs use the same ID and record question length rather than question text. Query logs include stage latency, candidate/final passage counts, pages, scores, and error code where applicable.

## Metrics and Probes

- `/health` is a cheap process liveness check with no dependency calls.
- `/ready` checks Ollama reachability, required model availability, Qdrant reachability, and active-index metadata without running inference.
- `/metrics` exposes Prometheus text metrics for total, successful, failed, and insufficient-context queries plus total, embedding, retrieval, reranking, and generation latency summaries.

Example:

```bash
curl http://localhost:8000/metrics
```

## Configuration Reference

| Setting | Default | Purpose |
| --- | ---: | --- |
| `RETRIEVAL_CANDIDATE_COUNT` | `20` | Qdrant candidates before reranking. |
| `RETRIEVAL_TOP_K` | `5` | Final passages sent to the generator. |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.0` | Qdrant-side minimum cosine score. |
| `MINIMUM_EVIDENCE_SCORE` | `0.2` | Minimum best vector score needed to generate. |
| `MINIMUM_LEXICAL_OVERLAP` | `0.05` | Minimum query/evidence token overlap needed to generate. |
| `MINIMUM_EVIDENCE_PASSAGES` | `1` | Minimum usable packed passages. |
| `RERANKING_ENABLED` | `false` | Enable the local reranking stage; off by default based on measured results. |
| `RERANKER_MODEL` | `lexical-v1` | Lightweight reranker implementation. |
| `RERANKER_VECTOR_WEIGHT` | `0.65` | Vector contribution to the combined rank score. |
| `STREAMING_ENABLED` | `true` | Enable `/api/query/stream`. |
| `OLLAMA_CONNECT_TIMEOUT_SECONDS` | `5` | Ollama connection timeout. |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Ollama read/generation timeout. |
| `QDRANT_TIMEOUT_SECONDS` | `15` | Vector-store operation timeout. |
| `RETRY_COUNT` | `2` | Bounded retry count for eligible Ollama calls. |
| `RETRY_BACKOFF_SECONDS` | `0.5` | Initial exponential-backoff delay. |
| `LOG_LEVEL` | `INFO` | Structured application log level. |

All supported settings and safe defaults are listed in `.env.example`.

## RAG Evaluation

`evaluation/dataset.jsonl` contains 23 questions derived from actual pages in the indexed HPE book: definitions, direct facts, comparisons, 2D/3D/monocular methods, heatmaps, regression, architectures, challenges, multi-page synthesis, and three deliberately unsupported questions. Expected answers are concise paraphrased key points rather than copied book passages.

The runner calculates Hit Rate@K, expected-page Recall@K, MRR, key-point coverage, citation presence, citation-page correctness, whether citations came from retrieved passages, and unsupported-question rejection. It evaluates vector-only and reranked retrieval in the same run and reports the measured delta without assuming improvement.

Run the full local evaluation against running Ollama and Qdrant services:

```bash
python -m evaluation.run
```

For the faster retrieval comparison without 23 generations:

```bash
python -m evaluation.run --retrieval-only
```

Reports are written to `evaluation/reports/latest.json` and `evaluation/reports/latest.md`. Full evaluation remains a documented local/manual check rather than a standard CI job because downloading and running the local models on every push would make CI slow and unreliable.

The latest measured 23-question local run is recorded in `evaluation/reports/latest.md`: vector and reranked Hit Rate@5 were both 1.000; vector Recall@5/MRR were 0.779/0.883, while reranked Recall@5/MRR were 0.708/0.864. Generation key-point coverage was 0.692, all answerable responses had citations, all citations came from retrieved passages and included an expected page, and all three unsupported questions were rejected. These results are why reranking is available but disabled by default.

## Example Query and Response

Example question:

> Why is monocular 3D human pose estimation considered ill-posed?

A representative shortened response has this form (the page is populated from the chunks actually retrieved from your indexed copy):

> A single 2D image does not uniquely determine a 3D pose: different depths and 3D body configurations can produce similar 2D projections. Recovering 3D pose therefore requires additional learned priors or constraints. `[p. <retrieved page>]`

The UI then lists the supporting source entry with its page, detected section, relevance score, and a short excerpt from the retrieved chunk. This example paraphrases the expected answer shape and does not reproduce book text or claim a fixed page number.

## Screenshots

No repository screenshot is currently available. Place a real application capture at `docs/images/application.png`, then replace this note with:

```markdown
![Human Pose Estimation Book Assistant](docs/images/application.png)
```

The `docs/images/` directory is tracked with a placeholder so screenshots can be added later without fabricating UI output.

## Project Structure

```text
.
|-- .github/workflows/ci.yml   # Backend and frontend continuous integration
|-- backend/
|   |-- app/                   # FastAPI, adapters, prompts, and focused services
|   |-- tests/                 # Unit, API, integration-boundary, and failure tests
|   `-- pyproject.toml         # Python dependencies and Ruff/Pytest configuration
|-- data/source/               # Ignored location for the local source PDF
|-- docs/images/               # Real UI screenshots when available
|-- evaluation/                # Book-grounded dataset, metrics, runner, and reports
|-- frontend/
|   |-- src/                   # React application and Vitest tests
|   `-- package.json           # Frontend scripts and dependencies
|-- .env.example               # Public configuration template
|-- docker-compose.yml         # Ollama, Qdrant, backend, and frontend services
`-- Makefile                   # Common local commands
```

## Testing

The fast checks do not require Ollama, Qdrant, a GPU, or the source PDF. Backend tests use fakes or mocked HTTP responses for external services and cover pipeline boundaries, API contracts, failure mapping, reranking fallback, streaming, persistence configuration, and the former Nginx 504 configuration.

Run the same backend commands used by CI:

```bash
cd backend
python -m pip install -e ".[dev]"
python -m ruff check . ../evaluation
python -m pytest
```

Run the same frontend commands used by CI:

```bash
cd frontend
npm ci
npm run lint
npm test
npm run build
```

The production build runs `tsc -b` before Vite, so TypeScript static checking is part of the build. You can also validate the resolved Docker configuration from the repository root with:

```bash
docker compose --env-file .env.example config --quiet
```

## Limitations

- CPU inference can be relatively slow; generation latency depends heavily on host hardware.
- The lightweight 1.5B generation model may produce weaker answers than larger models.
- The dependency-free reranker is much lighter than a cross-encoder but may be weaker on paraphrases; its impact must be judged from the generated evaluation report.
- Score and lexical evidence thresholds need calibration if the source, embedding model, or chunking strategy changes.
- PDF extraction is text-based. Scanned or image-only pages require OCR, which is not implemented.
- Diagrams, equations, complex layouts, and multi-column reading order may not survive PDF text extraction accurately.
- Chunking is character-based rather than token-based, with heuristic boundary and heading detection.
- The current configuration is designed around one primary source book rather than a multi-document library.

## Future Improvements

- Adopt a stronger generation model or optional cross-encoder reranker when suitable GPU/RAM resources are available and evaluation demonstrates a benefit.
- Add optional, documented GPU acceleration.
- Add OCR and layout-aware extraction for scanned or complex pages.
- Support multiple books and document management.

## Data and Reset Safety

The PDF is mounted read-only, and downloaded models and vectors live in named Docker volumes. `docker compose down` preserves them. The following command is intentionally destructive and removes both named volumes:

```bash
docker compose down -v
```

It does not remove the host PDF, but it deletes downloaded models and indexed vectors and should only be used for a full reset.

<!-- write-test -->
