from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SentenceCitation:
    """Evidencia asignada a una oración de la respuesta (post-generación)."""

    sentence: str
    chunk_id: str
    source: str
    page: int | None
    similarity_score: float


@dataclass(slots=True)
class Answer:
    """Respuesta con citas por oración, grounding agregado y metadatos de prompt."""

    question: str
    text: str
    citations: list[SentenceCitation] = field(default_factory=list)
    grounding_score: float | None = None
    hallucination_flag: bool = False
    hallucination_risk: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
