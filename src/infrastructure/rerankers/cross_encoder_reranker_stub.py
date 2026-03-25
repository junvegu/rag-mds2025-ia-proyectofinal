from src.application.ports.reranker_port import RerankerPort
from src.domain.entities.retrieval import HybridChunkResult, RerankedChunkResult


class CrossEncoderRerankerStub(RerankerPort):
    """Stub cross-encoder reranker adapter."""

    def rerank(
        self,
        query: str,
        candidates: list[HybridChunkResult],
        top_k: int,
    ) -> list[RerankedChunkResult]:
        raise NotImplementedError("Reranking is not implemented yet.")
