from src.application.ports.chunker_port import ChunkerPort
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document


class OverlapChunkerStub(ChunkerPort):
    """Stub chunker with overlap contract."""

    def chunk(self, document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        raise NotImplementedError("Chunking is not implemented yet.")
