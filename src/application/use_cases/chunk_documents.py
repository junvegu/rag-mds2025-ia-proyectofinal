from dataclasses import dataclass

from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker


@dataclass(slots=True)
class ChunkDocumentsUseCase:
    """Transform normalized documents into chunks with overlap."""

    chunker: RecursiveChunker

    def execute(
        self,
        documents: list[Document],
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_length: int = 120,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(
                self.chunker.chunk(
                    document=document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    min_chunk_length=min_chunk_length,
                )
            )
        return chunks
