from dataclasses import dataclass

from src.domain.entities.chunk import Chunk


@dataclass(slots=True)
class RetrievedChunk:
    """Chunk plus retrieval scores for hybrid retrieval pipelines."""

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
