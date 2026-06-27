from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from fastembed.sparse import SparseTextEmbedding

# BM25 — keyword-based, no neural network, ~10× faster than SPLADE.
# Retrieval quality is slightly lower for semantic queries but the
# cross-encoder reranker compensates for that.
_SPARSE_MODEL_NAME = "Qdrant/bm25"
_UPSERT_BATCH = 256   # points per Qdrant upsert call
_SPARSE_BATCH = 128   # texts per BM25 embed call

_sparse_model: SparseTextEmbedding | None = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=_SPARSE_MODEL_NAME)
    return _sparse_model


def _sparse_embed(texts: list[str]) -> list[SparseVector]:
    model = _get_sparse_model()
    results: list[SparseVector] = []
    for i in range(0, len(texts), _SPARSE_BATCH):
        batch = texts[i : i + _SPARSE_BATCH]
        for r in model.embed(batch):
            results.append(SparseVector(indices=r.indices.tolist(), values=r.values.tolist()))
    return results


class QdrantStorage:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "docs", dim: int = 768):
        self.client = QdrantClient(url=url, timeout=60)
        self.collection = collection
        self._ensure_collection(dim)

    def _ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            has_sparse = bool(getattr(info.config.params, "sparse_vectors", None))
            if not has_sparse:
                self.client.delete_collection(self.collection)
            else:
                return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
        )

    def upsert(self, ids: list[str], dense_vectors: list[list[float]], payloads: list[dict], texts: list[str]) -> None:
        sparse_vecs = _sparse_embed(texts)

        # Build all points then upsert in batches
        points = [
            PointStruct(
                id=ids[i],
                vector={"dense": dense_vectors[i], "sparse": sparse_vecs[i]},
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]
        for i in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + _UPSERT_BATCH],
            )

    def search(self, query_text: str, query_dense_vec: list[float], top_k: int = 10) -> dict:
        sparse_query = _sparse_embed([query_text])[0]

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=query_dense_vec, using="dense", limit=top_k * 3),
                Prefetch(query=sparse_query, using="sparse", limit=top_k * 3),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        contexts: list[str] = []
        sources: set[str] = set()
        for r in results.points:
            payload = r.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                if source:
                    sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}

    def delete_by_source(self, source_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source_id))]
                )
            ),
        )
