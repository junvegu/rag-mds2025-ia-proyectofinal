from src.application.ports.document_loader_port import DocumentLoaderPort
from src.domain.entities.document import Document


class LocalDocumentLoaderStub(DocumentLoaderPort):
    """Stub loader for local development wiring."""

    def load(self, source: str) -> list[Document]:
        raise NotImplementedError("Document loading is not implemented yet.")
