import logging
import re
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from src.application.ports.document_loader_port import DocumentLoaderPort
from src.domain.entities.document import Document

logger = logging.getLogger(__name__)


class WebLoader(DocumentLoaderPort):
    """Load and clean useful text from HTML pages."""

    def __init__(self, timeout_seconds: int = 15) -> None:
        self._timeout_seconds = timeout_seconds

    def load(self, source: str) -> list[Document]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError("requests and beautifulsoup4 are required to load web documents.") from exc

        try:
            response = requests.get(source, timeout=self._timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"Unable to fetch URL '{source}': {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = (soup.title.string.strip() if soup.title and soup.title.string else None) or self._fallback_title(source)
        text = self._normalize_whitespace(soup.get_text(separator=" "))
        if not text:
            logger.warning("Fetched URL has no useful text: %s", source)
            return []

        document = Document(
            doc_id=str(uuid5(NAMESPACE_URL, source)),
            title=title,
            text=text,
            source="sunat_web",
            page=None,
            metadata={"url": source},
        )
        return [document]

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _fallback_title(url: str) -> str:
        parsed = urlparse(url)
        return parsed.path.strip("/").split("/")[-1] or parsed.netloc
