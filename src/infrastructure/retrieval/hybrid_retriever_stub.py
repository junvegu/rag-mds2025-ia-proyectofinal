from src.application.ports.retriever_port import RetrieverPort
from src.domain.entities.retrieval import RetrievedChunk


class HybridRetrieverStub(RetrieverPort):
    """Stub hybrid retriever (BM25 + dense)."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError("Hybrid retrieval is not implemented yet.")
