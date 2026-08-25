from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evaluation.metrics import aggregate_generation, key_point_coverage, retrieval_scores

ROOT = Path(__file__).resolve().parents[1]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def post(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(path, json=payload)
    response.raise_for_status()
    return response.json()


async def evaluate(retrieval_only: bool, base_url: str) -> dict[str, Any]:
    samples = load_dataset(ROOT / "evaluation" / "dataset.jsonl")
    results: list[dict[str, Any]] = []
    timeout = httpx.Timeout(600, connect=10)
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        for sample in samples:
            baseline = await post(
                client,
                "/api/evaluation/retrieve",
                {"question": sample["question"], "rerank": False},
            )
            reranked = await post(
                client,
                "/api/evaluation/retrieve",
                {"question": sample["question"], "rerank": True},
            )
            item: dict[str, Any] = {
                "sample": sample,
                "baseline_pages": baseline["pages"],
                "reranked_pages": reranked["pages"],
                "baseline_scores": baseline["scores"],
                "reranked_scores": reranked["scores"],
            }
            if not retrieval_only:
                response = await post(client, "/api/query", {"question": sample["question"]})
                citation_pages = [citation["page"] for citation in response["citations"]]
                retrieved_pages = [chunk["page"] for chunk in response["retrieved_chunks"]]
                expected = set(sample["expected_pages"])
                response_summary = {
                    "status": response["status"],
                    "answer": response["answer"],
                    "citation_pages": citation_pages,
                    "retrieved_pages": retrieved_pages,
                    "timing": response["timing"],
                    "answerable": response["answerable"],
                    "confidence": response["confidence"],
                    "insufficient_context": response["insufficient_context"],
                    "reranking_applied": response["reranking_applied"],
                }
                item.update(
                    {
                        "response": response_summary,
                        "key_point_coverage": key_point_coverage(
                            response["answer"], sample["expected_answer_points"]
                        ),
                        "citation_pages": citation_pages,
                        "citation_page_correct": (
                            bool(expected & set(citation_pages)) if expected else True
                        ),
                        "citations_from_retrieval": set(citation_pages).issubset(retrieved_pages),
                        "unsupported_rejected": (
                            response["insufficient_context"]
                            if not sample["answerable_from_book"]
                            else False
                        ),
                    }
                )
            results.append(item)
    baseline_scores = retrieval_scores(results, "baseline_pages")
    reranked_scores = retrieval_scores(results, "reranked_pages")
    generation = aggregate_generation(results)
    overall_parts = [reranked_scores.hit_rate, reranked_scores.recall, reranked_scores.mrr]
    if not retrieval_only:
        overall_parts.extend(generation.values())
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "sample_count": len(samples),
        "retrieval_only": retrieval_only,
        "baseline_retrieval": baseline_scores.__dict__,
        "reranked_retrieval": reranked_scores.__dict__,
        "reranker_delta": {
            key: getattr(reranked_scores, key) - getattr(baseline_scores, key)
            for key in ("hit_rate", "recall", "mrr")
        },
        "generation": generation,
        "overall_score": sum(overall_parts) / len(overall_parts),
        "failed_cases": [
            item["sample"]["id"]
            for item in results
            if item["sample"]["answerable_from_book"]
            and not (set(item["sample"]["expected_pages"]) & set(item["reranked_pages"]))
        ],
        "results": results,
    }


def markdown_report(report: dict[str, Any]) -> str:
    baseline = report["baseline_retrieval"]
    reranked = report["reranked_retrieval"]
    delta = report["reranker_delta"]
    generation = report["generation"]
    lines = [
        "# HPE RAG Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Samples: {report['sample_count']}",
        f"Overall score: {report['overall_score']:.3f}",
        "",
        "## Retrieval",
        "",
        "| Metric | Vector baseline | Reranked | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("hit_rate", "recall", "mrr"):
        lines.append(
            f"| {key} | {baseline[key]:.3f} | {reranked[key]:.3f} | {delta[key]:+.3f} |"
        )
    lines.extend(("", "## Generation and grounding", ""))
    if report["retrieval_only"]:
        lines.append("Not run (retrieval-only mode).")
    else:
        for key, value in generation.items():
            lines.append(f"- {key}: {value:.3f}")
    lines.extend(("", "## Failed retrieval cases", ""))
    lines.append(", ".join(report["failed_cases"]) or "None")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval, grounding, and generation")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip slow local generation while still comparing retrieval and reranking",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "reports")
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.retrieval_only, args.base_url))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "latest.md").write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
