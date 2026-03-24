from abc import ABC, abstractmethod

from src.domain.entities.retrieval import RetrievedChunk


class RetrieverPort(ABC):
    """Retrieve candidates using dense, sparse, or hybrid strategies."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return top retrieval candidates."""
        raise NotImplementedError
