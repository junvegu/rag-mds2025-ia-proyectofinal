from abc import ABC, abstractmethod


class EmbeddingPort(ABC):
    """Embed texts and queries into dense vectors."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts for indexing."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a user query for dense retrieval."""
        raise NotImplementedError
