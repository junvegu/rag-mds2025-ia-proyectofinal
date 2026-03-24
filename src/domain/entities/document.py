from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """Immutable-like document model from any academic source."""

    document_id: str
    content: str
    source: str
    title: str | None = None
    source_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)
