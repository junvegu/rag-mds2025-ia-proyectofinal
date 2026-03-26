from dataclasses import dataclass, field
import logging
from pathlib import Path
from urllib.parse import urlparse

from src.domain.entities.document import Document
from src.infrastructure.loaders.http_headers import browser_like_headers
from src.infrastructure.loaders.pdf_loader import PDFLoader
from src.infrastructure.loaders.txt_loader import TXTLoader
from src.infrastructure.loaders.web_loader import WebLoader

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class URLDatasetLoader:
    """Coordinate source detection and loader dispatching."""

    pdf_loader: PDFLoader = field(default_factory=PDFLoader)
    txt_loader: TXTLoader = field(default_factory=TXTLoader)
    web_loader: WebLoader = field(default_factory=WebLoader)

    def load(self, sources: list[str]) -> list[Document]:
        documents: list[Document] = []
        for source in sources:
            try:
                documents.extend(self._load_one(source))
            except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
                logger.warning("Failed to load source '%s': %s", source, exc)
        return documents

    def _load_one(self, source: str) -> list[Document]:
        if self._is_url(source):
            url_kind = self._detect_remote_kind(source)
            return self.pdf_loader.load_from_url(source) if url_kind == "pdf" else self.web_loader.load(source)

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")
        if source_path.is_dir():
            docs: list[Document] = []
            docs.extend(self.pdf_loader.load(str(source_path)))
            docs.extend(self.txt_loader.load(str(source_path)))
            return docs

        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            return self.pdf_loader.load(source)
        if suffix == ".txt":
            return self.txt_loader.load(source)
        raise ValueError(f"Unsupported source format: {source}")

    @staticmethod
    def _is_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _detect_remote_kind(self, url: str) -> str:
        """Detect whether remote URL is likely PDF or HTML."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        if path.endswith(".pdf") or ".pdf?" in url.lower():
            return "pdf"

        try:
            import requests
        except ImportError:
            return "html"

        try:
            response = requests.head(
                url,
                timeout=10,
                allow_redirects=True,
                headers=browser_like_headers(for_pdf=path.endswith(".pdf")),
            )
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type:
                return "pdf"
            if "text/html" in content_type:
                return "html"
        except requests.RequestException:
            logger.info("HEAD request failed for '%s'; falling back to extension heuristics.", url)

        return "pdf" if "pdf" in path else "html"
