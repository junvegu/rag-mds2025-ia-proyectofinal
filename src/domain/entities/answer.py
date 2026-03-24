from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Citation:
    """Grounding reference attached to an answer sentence."""

    chunk_id: str
    document_id: str
    sentence_index: int | None = None
    score: float | None = None
    excerpt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Answer:
    """Answer model with sentence-level citations and quality metadata."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounding_score: float | None = None
    hallucination_risk: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
