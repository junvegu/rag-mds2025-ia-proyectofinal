from dataclasses import dataclass

from src.domain.entities.retrieval import RetrievedChunk
from src.infrastructure.vectorstores.faiss_hnsw_store import FAISSHNSWStore


@dataclass(slots=True)
class SearchVectorIndexUseCase:
    """Search nearest chunks from a vector index."""

    vector_store: FAISSHNSWStore

    def execute(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        return self.vector_store.search(query_embedding=query_embedding, top_k=top_k)
