from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models


class DimensionMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, url: str, collection: str, timeout: float) -> None:
        self.client = AsyncQdrantClient(url=url, timeout=timeout)
        self.collection = collection

    async def close(self) -> None:
        await self.client.close()

    async def exists(self, name: str | None = None) -> bool:
        return await self.client.collection_exists(name or self.collection)

    async def collection_dimension(self, name: str | None = None) -> int | None:
        target = name or self.collection
        if not await self.exists(target):
            return None
        info = await self.client.get_collection(target)
        vectors = info.config.params.vectors
        return int(vectors.size) if hasattr(vectors, "size") else None

    async def create(self, name: str, dimension: int) -> None:
        await self.client.create_collection(
            name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    async def delete(self, name: str) -> None:
        if await self.exists(name):
            await self.client.delete_collection(name)

    async def upsert(self, name: str, points: list[models.PointStruct]) -> None:
        await self.client.upsert(name, points=points, wait=True)

    async def activate(self, temporary: str) -> None:
        aliases = await self.client.get_aliases()
        active = next(
            (
                item.collection_name
                for item in aliases.aliases
                if item.alias_name == self.collection
            ),
            None,
        )
        operations: list[models.AliasOperations] = []
        if active:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.collection)
                )
            )
        elif await self.client.collection_exists(self.collection):
            raise RuntimeError(
                "A legacy physical collection blocks atomic alias activation; "
                "remove it or choose a new COLLECTION_NAME"
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=temporary, alias_name=self.collection
                )
            )
        )
        await self.client.update_collection_aliases(change_aliases_operations=operations)
        if active and active != temporary:
            await self.delete(active)

    async def search(self, vector: list[float], limit: int, threshold: float) -> list[SearchHit]:
        dimension = await self.collection_dimension()
        if dimension is None:
            return []
        if dimension != len(vector):
            raise DimensionMismatchError(
                f"Index dimension {dimension} does not match embedding dimension "
                f"{len(vector)}; force re-indexing is required"
            )
        response = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        )
        return [
            SearchHit(str(point.id), float(point.score), dict(point.payload or {}))
            for point in response.points
        ]

    async def metadata(self) -> dict[str, Any] | None:
        if not await self.exists():
            return None
        records, _ = await self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="record_type", match=models.MatchValue(value="manifest")
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return dict(records[0].payload or {}) if records else None
