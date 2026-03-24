from src.application.ports.embedding_port import EmbeddingPort


class EmbeddingModelStub(EmbeddingPort):
    """Stub embedding adapter."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Text embedding is not implemented yet.")

    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError("Query embedding is not implemented yet.")
