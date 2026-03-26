from pathlib import Path
import logging
import tempfile

from src.domain.entities.document import Document
from src.infrastructure.loaders.common import build_stable_doc_id, infer_title_from_source
from src.infrastructure.loaders.http_headers import browser_like_headers
from src.infrastructure.loaders.text_utils import normalize_text

logger = logging.getLogger(__name__)


class PDFLoader:
    """Load PDF content from local paths or URLs."""

    def __init__(self, min_text_length: int = 40, timeout_seconds: int = 20) -> None:
        self._min_text_length = min_text_length
        self._timeout_seconds = timeout_seconds

    def load(self, path: str) -> list[Document]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"PDF path not found: {source}")

        files = [source] if source.is_file() else sorted(source.glob("*.pdf"))
        documents: list[Document] = []
        for file_path in files:
            if file_path.suffix.lower() != ".pdf":
                continue
            documents.extend(self._load_single_pdf(file_path=file_path, source=str(file_path)))
        return documents

    def load_from_url(self, url: str) -> list[Document]:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests is required for loading PDFs from URL.") from exc

        try:
            response = requests.get(
                url,
                timeout=self._timeout_seconds,
                headers=browser_like_headers(for_pdf=True),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"Could not download PDF URL '{url}': {exc}") from exc

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            return self._load_single_pdf(file_path=Path(tmp_file.name), source=url)

    def _load_single_pdf(self, file_path: Path, source: str) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf is required to parse PDF files.") from exc

        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid or unreadable PDF '{source}': {exc}") from exc

        title = self._extract_title(source=source, reader=reader)
        documents: list[Document] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if len(text) < self._min_text_length:
                continue
            documents.append(
                Document(
                    doc_id=build_stable_doc_id(f"{source}#page={page_idx}"),
                    title=title,
                    text=text,
                    source=source,
                    page=page_idx,
                    metadata={
                        "source_type": "url_pdf" if source.startswith("http") else "local_pdf",
                        "original_source": source,
                    },
                )
            )
        return documents

    @staticmethod
    def _extract_title(source: str, reader: object) -> str:
        metadata_title = getattr(getattr(reader, "metadata", None), "title", None)
        if metadata_title:
            return str(metadata_title).strip()
        return infer_title_from_source(source, fallback="untitled_pdf")
