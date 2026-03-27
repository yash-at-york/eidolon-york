"""
Eidolon - Qdrant Vector Store Client
Wraps the Qdrant Python client for ghost node storage and semantic search.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ghost_config import (
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_IN_MEMORY,
    QDRANT_PORT,
    QDRANT_VECTOR_DIM,
)


class VectorStore:
    """
    Qdrant-backed vector store for ghost node embeddings.

    Each point represents a CPG node:
      - id:      hash of the node_id string (Qdrant requires an integer or UUID)
      - vector:  256-dim CodeT5+ embedding of the node's structural JSON
      - payload: {"node_id": "h_7b9x", "service": "svc", "type": "Function", ...}
    """

    def __init__(self) -> None:
        if QDRANT_IN_MEMORY:
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=QDRANT_VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )

    def _node_id_to_int(self, node_id: str) -> int:
        """Convert 'h_XXXXXXXX' to a stable integer point ID for Qdrant."""
        return int(node_id.replace("h_", ""), 16) if node_id.startswith("h_") else hash(node_id) & 0xFFFFFFFF

    def upsert(self, node_id: str, vector: list[float], metadata: dict) -> None:
        """Insert or update a single ghost node embedding."""
        payload = {"node_id": node_id, **metadata}
        point = PointStruct(
            id=self._node_id_to_int(node_id),
            vector=vector,
            payload=payload,
        )
        self._client.upsert(collection_name=QDRANT_COLLECTION, points=[point])

    def upsert_batch(self, records: list[tuple[str, list[float], dict]]) -> None:
        """Batch upsert for efficiency during delta sync."""
        points = [
            PointStruct(
                id=self._node_id_to_int(node_id),
                vector=vector,
                payload={"node_id": node_id, **metadata},
            )
            for node_id, vector, metadata in records
        ]
        if points:
            self._client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        service_filter: str | None = None,
    ) -> list[dict]:
        """
        Semantic similarity search.
        Returns list of dicts with {node_id, score, service, type, ...}.
        """
        qdrant_filter = None
        if service_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(key="service", match=MatchValue(value=service_filter))]
            )

        response = self._client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        results = response.points

        return [
            {"score": hit.score, **hit.payload}
            for hit in results
        ]

    def delete_by_service(self, service: str) -> None:
        """Delete all ghost nodes for a given service (useful for re-sync)."""
        self._client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="service", match=MatchValue(value=service))]
            ),
        )
