.PHONY: up down logs ingest test lint build config evaluate evaluate-retrieval
up:
	docker compose up --build -d
down:
	docker compose down
logs:
	docker compose logs -f
ingest:
	curl -X POST http://localhost:8000/api/ingest -H "Content-Type: application/json" -d '{"force":false}'
test:
	cd backend && python -m pytest
	cd frontend && npm test
lint:
	cd backend && ruff check .
	cd frontend && npm run lint
build:
	cd frontend && npm run build
config:
	docker compose config
evaluate:
	python -m evaluation.run
evaluate-retrieval:
	python -m evaluation.run --retrieval-only
