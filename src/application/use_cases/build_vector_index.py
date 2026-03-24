from dataclasses import dataclass

from src.application.use_cases.embed_chunks import EmbeddedChunk
from src.infrastructure.vectorstores.faiss_hnsw_store import ChunkMetadata, FAISSHNSWStore


@dataclass(slots=True)
class BuildVectorIndexUseCase:
    """Build and optionally persist FAISS HNSW index from embedded chunks."""

    vector_store: FAISSHNSWStore

    def execute(self, embedded_chunks: list[EmbeddedChunk], save_dir: str | None = None) -> FAISSHNSWStore:
        embeddings = [item.embedding for item in embedded_chunks]
        metadata = [
            ChunkMetadata(
                chunk_id=item.chunk_id,
                doc_id=item.doc_id,
                source=item.source,
                page=item.page,
                text=item.text,
            )
            for item in embedded_chunks
        ]
        self.vector_store.add(embeddings=embeddings, metadata=metadata)
        if save_dir is not None:
            self.vector_store.save(save_dir)
        return self.vector_store
