from dataclasses import dataclass

from src.application.ports.reranker_port import RerankerPort
from src.domain.entities.retrieval import HybridChunkResult, RerankedChunkResult


@dataclass(slots=True)
class RerankContextUseCase:
    """Second stage after hybrid retrieval: cross-encoder reordering."""

    reranker: RerankerPort

    def execute(
        self,
        query: str,
        candidates: list[HybridChunkResult],
        top_k: int,
    ) -> list[RerankedChunkResult]:
        """
        Rerank hybrid candidates and return at most ``top_k`` rows with ``rerank_position`` 1..k.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        return self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
