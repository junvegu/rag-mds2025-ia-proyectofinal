from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

from src.application.ports.vector_store_port import VectorStorePort
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkMetadata:
    """Minimal metadata stored alongside vector rows."""

    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    text: str


class FAISSHNSWStore(VectorStorePort):
    """FAISS HNSW store for dense vector indexing and search."""

    def __init__(
        self,
        dimension: int = 384,
        m: int = 24,
        ef_construction: int = 120,
        ef_search: int = 48,
    ) -> None:
        self._dimension = dimension
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._index = self._create_index()
        self._metadata: list[ChunkMetadata] = []

    @property
    def vector_count(self) -> int:
        """Number of vectors currently indexed."""
        return int(self._index.ntotal)

    @property
    def metadata_count(self) -> int:
        """Number of metadata rows (should match vector_count when consistent)."""
        return len(self._metadata)

    def add(self, embeddings: list[list[float]], metadata: list[ChunkMetadata]) -> None:
        """Add embeddings and metadata in aligned order."""
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata length must match.")
        if not embeddings:
            return

        matrix = self._to_float32_matrix(embeddings)
        if matrix.shape[1] != self._dimension:
            raise ValueError(f"Expected embedding dimension {self._dimension}, got {matrix.shape[1]}.")

        self._index.add(matrix)
        self._metadata.extend(metadata)
        logger.info("Added %s vectors to HNSW index.", len(embeddings))

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Compatibility helper with existing port naming."""
        metadata = [
            ChunkMetadata(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                page=chunk.page,
                text=chunk.text,
            )
            for chunk in chunks
        ]
        self.add(embeddings=embeddings, metadata=metadata)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return nearest neighbors with chunk reconstruction."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if self._index.ntotal == 0:
            return []

        query = self._to_float32_matrix([query_embedding])
        if query.shape[1] != self._dimension:
            raise ValueError(f"Expected query dimension {self._dimension}, got {query.shape[1]}.")

        scores, indices = self._index.search(query, top_k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            chunk = Chunk(
                chunk_id=meta.chunk_id,
                doc_id=meta.doc_id,
                text=meta.text,
                source=meta.source,
                page=meta.page,
                metadata={"source": meta.source, "page": meta.page},
            )
            results.append(RetrievedChunk(chunk=chunk, dense_score=float(score), final_score=float(score)))
        return results

    def save(self, directory: str) -> None:
        """Persist FAISS index and metadata to disk."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        index_path = target / "faiss_hnsw.index"
        metadata_path = target / "faiss_hnsw_metadata.json"

        faiss = self._faiss()
        faiss.write_index(self._index, str(index_path))
        payload = {
            "dimension": self._dimension,
            "m": self._m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "metadata": [asdict(item) for item in self._metadata],
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved FAISS HNSW index to %s", target)

    @classmethod
    def load(cls, directory: str) -> FAISSHNSWStore:
        """Load FAISS index and metadata from disk."""
        source = Path(directory)
        index_path = source / "faiss_hnsw.index"
        metadata_path = source / "faiss_hnsw_metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index files not found in '{source}'.")

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        store = cls(
            dimension=int(payload["dimension"]),
            m=int(payload["m"]),
            ef_construction=int(payload["ef_construction"]),
            ef_search=int(payload["ef_search"]),
        )
        faiss = store._faiss()
        store._index = faiss.read_index(str(index_path))
        store._metadata = [ChunkMetadata(**item) for item in payload.get("metadata", [])]
        store._index.hnsw.efSearch = store._ef_search
        return store

    def _create_index(self) -> object:
        faiss = self._faiss()
        index = faiss.IndexHNSWFlat(self._dimension, self._m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search
        return index

    @staticmethod
    def _faiss() -> object:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required for FAISS HNSW indexing.") from exc
        return faiss

    @staticmethod
    def _to_float32_matrix(vectors: list[list[float]]) -> object:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for FAISS vector preparation.") from exc
        return np.asarray(vectors, dtype="float32")
