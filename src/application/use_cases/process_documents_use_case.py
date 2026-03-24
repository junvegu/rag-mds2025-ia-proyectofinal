from dataclasses import dataclass

from src.application.ports.chunker_port import ChunkerPort
from src.application.ports.document_loader_port import DocumentLoaderPort
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document


@dataclass(slots=True)
class ProcessDocumentsResult:
    """Result object for document loading and chunking."""

    documents: list[Document]
    chunks: list[Chunk]


@dataclass(slots=True)
class ProcessDocumentsUseCase:
    """Load text files and transform them into chunks."""

    loader: DocumentLoaderPort
    chunker: ChunkerPort

    def execute(self, source_dir: str, chunk_size: int, chunk_overlap: int) -> ProcessDocumentsResult:
        documents = self.loader.load(source=source_dir)
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(
                self.chunker.chunk(
                    document=document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        return ProcessDocumentsResult(documents=documents, chunks=chunks)
