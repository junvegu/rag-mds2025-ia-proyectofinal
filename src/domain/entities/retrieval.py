from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.domain.entities.chunk import Chunk

ScoreNormalizationMode = Literal["softmax", "minmax"]


@dataclass(slots=True)
class RetrievedChunk:
    """Chunk plus retrieval scores for hybrid retrieval pipelines."""

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None


@dataclass(slots=True)
class HybridRetrieverConfig:
    """SUNAT-oriented hybrid: softmax (default) or min-max per channel, then weighted sum."""

    # Defaults: ligero sesgo léxico (0.55) para desambiguar categorías tributarias similares
    # sin dejar de privilegiar el denso cuando la pregunta es parafrástica.
    dense_weight: float = 0.45
    sparse_weight: float = 0.55
    top_k_dense: int = 20
    top_k_sparse: int = 20
    final_top_k: int = 10
    score_normalization: ScoreNormalizationMode = "softmax"
    fusion_temperature: float = 1.0


@dataclass(slots=True)
class HybridChunkResult:
    """Single chunk after hybrid scoring (ready for reranking)."""

    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    text: str
    dense_score: float | None
    sparse_score: float | None
    hybrid_score: float


@dataclass(slots=True)
class RerankedChunkResult:
    """Hybrid metadata preserved plus cross-encoder score and final rank (1-based)."""

    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    text: str
    dense_score: float | None
    sparse_score: float | None
    hybrid_score: float
    rerank_score: float
    rerank_position: int

    @classmethod
    def from_hybrid(cls, h: HybridChunkResult, rerank_score: float, rerank_position: int) -> RerankedChunkResult:
        return cls(
            chunk_id=h.chunk_id,
            doc_id=h.doc_id,
            source=h.source,
            page=h.page,
            text=h.text,
            dense_score=h.dense_score,
            sparse_score=h.sparse_score,
            hybrid_score=h.hybrid_score,
            rerank_score=rerank_score,
            rerank_position=rerank_position,
        )
