from abc import ABC, abstractmethod

from src.domain.entities.document import Document


class DocumentLoaderPort(ABC):
    """Load source documents from files or external locations."""

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Return normalized documents."""
        raise NotImplementedError
