from pathlib import Path
import logging

from src.domain.entities.document import Document
from src.infrastructure.loaders.common import build_stable_doc_id
from src.infrastructure.loaders.text_utils import normalize_text

logger = logging.getLogger(__name__)


class TXTLoader:
    """Load local TXT files into normalized Document objects."""

    def __init__(self, min_text_length: int = 40) -> None:
        self._min_text_length = min_text_length

    def load(self, path: str) -> list[Document]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"TXT path not found: {source}")

        files = [source] if source.is_file() else sorted(source.glob("*.txt"))
        documents: list[Document] = []

        for file_path in files:
            if file_path.suffix.lower() != ".txt":
                continue
            try:
                raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Could not read TXT file '%s': %s", file_path, exc)
                continue

            text = normalize_text(raw_text)
            if len(text) < self._min_text_length:
                logger.info("Skipping short TXT file '%s'.", file_path)
                continue

            documents.append(
                Document(
                    doc_id=build_stable_doc_id(str(file_path.resolve())),
                    title=file_path.stem,
                    text=text,
                    source=str(file_path),
                    page=None,
                    metadata={
                        "source_type": "local_txt",
                        "source_file": file_path.name,
                        "original_source": str(file_path),
                    },
                )
            )
        return documents
