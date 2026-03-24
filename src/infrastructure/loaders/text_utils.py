import re


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving readable paragraph flow."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


def normalize_web_text(text: str) -> str:
    """Normalize text extracted from HTML pages for RAG usage."""
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\u00a0", " ", cleaned)
    cleaned = re.sub(r"[^\S\n]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n ?\n+", "\n\n", cleaned)
    return cleaned.strip()
