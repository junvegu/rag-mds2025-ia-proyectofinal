import logging
import re
from urllib.parse import urlparse

from src.domain.entities.document import Document
from src.infrastructure.loaders.common import build_stable_doc_id, infer_title_from_source
from src.infrastructure.loaders.text_utils import normalize_web_text

logger = logging.getLogger(__name__)


class WebLoader:
    """Load useful text content from web pages."""

    def __init__(self, min_text_length: int = 80, timeout_seconds: int = 20) -> None:
        self._min_text_length = min_text_length
        self._timeout_seconds = timeout_seconds

    def load(self, url: str) -> list[Document]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError("requests and beautifulsoup4 are required for WebLoader.") from exc

        try:
            response = requests.get(url, timeout=self._timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"Could not fetch URL '{url}': {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        for tag_name in ("script", "style", "noscript", "nav", "header", "footer", "aside"):
            for tag in soup.find_all(tag_name):
                tag.decompose()
        # Remove frequent UI and cookie/banner blocks.
        for selector in (
            ".breadcrumb",
            ".breadcrumbs",
            ".menu",
            ".navbar",
            ".sidebar",
            ".share",
            ".social",
            ".cookie",
            "#cookie-banner",
            "#accessibility",
        ):
            for tag in soup.select(selector):
                tag.decompose()

        main = self._select_main_content(soup) or soup.body or soup
        raw_text = main.get_text(separator="\n")
        raw_text = re.sub(r"\n{2,}", "\n\n", raw_text)
        text = normalize_web_text(raw_text)
        if len(text) < self._min_text_length:
            logger.info("Skipping short or noisy web page '%s'.", url)
            return []

        title = self._extract_title(soup=soup, url=url)
        return [
            Document(
                doc_id=build_stable_doc_id(url),
                title=title,
                text=text,
                source=url,
                page=None,
                metadata={"source_type": "url_html", "original_source": url},
            )
        ]

    @staticmethod
    def _extract_title(soup: object, url: str) -> str:
        title_tag = getattr(soup, "title", None)
        if title_tag and getattr(title_tag, "string", None):
            return str(title_tag.string).strip()
        return infer_title_from_source(url, fallback=urlparse(url).netloc)

    @staticmethod
    def _select_main_content(soup: object) -> object | None:
        """Prefer content containers common in gob.pe and SUNAT pages."""
        selectors = (
            "main",
            "article",
            "[role='main']",
            ".page-content",
            ".contenido",
            ".content",
            "#contenido",
            "#content",
        )
        for selector in selectors:
            node = getattr(soup, "select_one", lambda _: None)(selector)
            if node is not None:
                return node
        return None
