from abc import ABC, abstractmethod

from src.domain.entities.retrieval import HybridChunkResult, RerankedChunkResult


class RerankerPort(ABC):
    """Rerank hybrid retrieval candidates with a cross-encoder or equivalent."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[HybridChunkResult],
        top_k: int,
    ) -> list[RerankedChunkResult]:
        """Return candidates sorted by relevance; at most top_k items."""
        raise NotImplementedError
