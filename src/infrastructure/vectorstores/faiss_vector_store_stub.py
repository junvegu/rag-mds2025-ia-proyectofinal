from src.application.ports.vector_store_port import VectorStorePort
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval import RetrievedChunk


class FaissVectorStoreStub(VectorStorePort):
    """Stub FAISS-like vector store adapter."""

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError("Vector index upsert is not implemented yet.")

    def save(self, directory: str) -> None:
        raise NotImplementedError("Vector index persistence is not implemented yet.")

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError("Dense vector search is not implemented yet.")
