from abc import ABC, abstractmethod

from src.domain.entities.retrieval import RetrievedChunk


class RerankerPort(ABC):
    """Rerank retrieval candidates with stronger relevance models."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Return reranked candidates."""
        raise NotImplementedError
