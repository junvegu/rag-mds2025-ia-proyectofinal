from pathlib import Path
import logging
from uuid import NAMESPACE_URL, uuid5

from src.application.ports.document_loader_port import DocumentLoaderPort
from src.domain.entities.document import Document

logger = logging.getLogger(__name__)


class TXTLoader(DocumentLoaderPort):
    """Load plain text documents from a file or directory."""

    def load(self, source: str) -> list[Document]:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"TXT source not found: {source_path}")

        files = [source_path] if source_path.is_file() else sorted(source_path.glob("*.txt"))
        documents: list[Document] = []

        for file_path in files:
            if file_path.suffix.lower() != ".txt":
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError as exc:
                logger.warning("Skipping unreadable TXT file '%s': %s", file_path, exc)
                continue

            if not text:
                logger.info("Skipping empty TXT file '%s'.", file_path)
                continue

            documents.append(
                Document(
                    doc_id=str(uuid5(NAMESPACE_URL, str(file_path.resolve()))),
                    title=file_path.stem,
                    text=text,
                    source="sunat_txt",
                    page=None,
                    metadata={"source_file": file_path.name, "path": str(file_path)},
                )
            )
        return documents
