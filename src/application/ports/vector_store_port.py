from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval import RetrievedChunk


class VectorStorePort(ABC):
    """Store and search dense vectors (e.g., FAISS indexes)."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunk vectors."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Run dense search and return scored candidates."""
        raise NotImplementedError
