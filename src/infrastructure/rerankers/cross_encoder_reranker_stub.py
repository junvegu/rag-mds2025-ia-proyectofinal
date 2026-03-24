from src.application.ports.reranker_port import RerankerPort
from src.domain.entities.retrieval import RetrievedChunk


class CrossEncoderRerankerStub(RerankerPort):
    """Stub cross-encoder reranker adapter."""

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError("Reranking is not implemented yet.")
