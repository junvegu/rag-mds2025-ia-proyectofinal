from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """Normalized source document used across ingestion pipelines."""

    doc_id: str
    title: str
    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        """Backward-compatible alias for previous field name."""
        return self.doc_id

    @property
    def content(self) -> str:
        """Backward-compatible alias for previous field name."""
        return self.text
