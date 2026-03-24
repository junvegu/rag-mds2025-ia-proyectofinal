from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from src.application.ports.document_loader_port import DocumentLoaderPort
from src.domain.entities.document import Document


class TextFileDocumentLoader(DocumentLoaderPort):
    """Load `.txt` files from a local directory."""

    def load(self, source: str) -> list[Document]:
        source_dir = Path(source)
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        if not source_dir.is_dir():
            raise ValueError(f"Source path is not a directory: {source_dir}")

        documents: list[Document] = []
        for file_path in sorted(source_dir.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            document_id = str(uuid5(NAMESPACE_URL, str(file_path.resolve())))
            documents.append(
                Document(
                    doc_id=document_id,
                    title=file_path.stem,
                    text=text,
                    source="sunat_txt",
                    page=None,
                    metadata={"source_file": file_path.name, "path": str(file_path)},
                )
            )
        return documents
