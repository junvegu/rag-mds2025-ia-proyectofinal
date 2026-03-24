from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document


class ChunkerPort(ABC):
    """Split documents into chunks with overlap."""

    @abstractmethod
    def chunk(self, document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        """Create ordered chunks ready for indexing."""
        raise NotImplementedError
