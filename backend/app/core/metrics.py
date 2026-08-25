from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        for name in (
            "rag_queries_total",
            "rag_queries_success_total",
            "rag_queries_failed_total",
            "rag_insufficient_context_total",
        ):
            self._counters[name] = 0
        for name in (
            "rag_query_latency_seconds",
            "rag_embedding_latency_seconds",
            "rag_retrieval_latency_seconds",
            "rag_reranking_latency_seconds",
            "rag_generation_latency_seconds",
        ):
            self._sums[name] = 0
            self._counts[name] = 0

    def increment(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    def observe_ms(self, name: str, value: float) -> None:
        with self._lock:
            self._sums[name] += value / 1000
            self._counts[name] += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                lines.extend((f"# TYPE {name} counter", f"{name} {value:g}"))
            for name, value in sorted(self._sums.items()):
                lines.extend(
                    (
                        f"# TYPE {name} summary",
                        f'{name}_sum {value:.6f}',
                        f'{name}_count {self._counts[name]}',
                    )
                )
        return "\n".join(lines) + "\n"


metrics = Metrics()
