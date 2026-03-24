from dataclasses import dataclass, field

from src.domain.entities.document import Document
from src.infrastructure.loaders.url_dataset_loader import URLDatasetLoader


@dataclass(slots=True)
class LoadDocumentsUseCase:
    """Application use case for document ingestion."""

    dataset_loader: URLDatasetLoader = field(default_factory=URLDatasetLoader)

    def execute(self, sources: list[str]) -> list[Document]:
        return self.dataset_loader.load(sources)
