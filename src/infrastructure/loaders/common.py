from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import NAMESPACE_URL, uuid5


def build_stable_doc_id(seed: str) -> str:
    """Create deterministic IDs for repeatable ingestion runs."""
    return str(uuid5(NAMESPACE_URL, seed))


def infer_title_from_source(source: str, fallback: str = "untitled") -> str:
    """Infer readable title from a local path or URL."""
    parsed = urlparse(source)
    raw_name = Path(unquote(parsed.path if parsed.scheme else source)).name
    return Path(raw_name).stem or fallback
