from dataclasses import dataclass

from src.application.ports.embedding_port import EmbeddingPort
from src.domain.entities.chunk import Chunk


@dataclass(slots=True)
class EmbeddedChunk:
    """Chunk plus dense vector, ready for vector indexing."""

    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    text: str
    embedding: list[float]


@dataclass(slots=True)
class EmbedChunksUseCase:
    """Generate embeddings for chunk collections."""

    embedding_adapter: EmbeddingPort

    def execute(self, chunks: list[Chunk], batch_size: int | None = None) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if batch_size is None:
            return self._embed_batch(chunks)

        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), batch_size):
            embedded.extend(self._embed_batch(chunks[start : start + batch_size]))
        return embedded

    def _embed_batch(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_adapter.embed_texts(texts)
        if len(vectors) != len(chunks):
            raise ValueError("Embedding adapter returned inconsistent vector count.")

        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                page=chunk.page,
                text=chunk.text,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors)
        ]
