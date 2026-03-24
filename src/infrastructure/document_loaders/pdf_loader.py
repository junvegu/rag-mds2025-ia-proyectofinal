from pathlib import Path
import logging
from uuid import NAMESPACE_URL, uuid5

from src.application.ports.document_loader_port import DocumentLoaderPort
from src.domain.entities.document import Document

logger = logging.getLogger(__name__)


class PDFLoader(DocumentLoaderPort):
    """Load PDF content and return one Document per page."""

    def load(self, source: str) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf is required to load PDF documents.") from exc

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"PDF source not found: {source_path}")

        files = [source_path] if source_path.is_file() else sorted(source_path.glob("*.pdf"))
        documents: list[Document] = []

        for file_path in files:
            if file_path.suffix.lower() != ".pdf":
                continue
            try:
                reader = PdfReader(str(file_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping unreadable PDF '%s': %s", file_path, exc)
                continue

            for index, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        doc_id=str(uuid5(NAMESPACE_URL, f"{file_path.resolve()}#page={index}")),
                        title=file_path.stem,
                        text=text,
                        source="sunat_pdf",
                        page=index,
                        metadata={"source_file": file_path.name, "path": str(file_path)},
                    )
                )
        return documents
