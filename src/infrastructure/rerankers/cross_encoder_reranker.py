"""Cross-encoder reranking via sentence-transformers."""

from __future__ import annotations

import logging

from src.application.ports.reranker_port import RerankerPort
from src.domain.entities.retrieval import HybridChunkResult, RerankedChunkResult

logger = logging.getLogger(__name__)


def _flatten_scores(raw: object) -> list[float]:
    """One score per pair: CrossEncoder may return (n,) or (n,2) logits."""
    import numpy as np

    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        arr = arr[:, -1]
    flat = np.atleast_1d(arr).reshape(-1)
    return [float(x) for x in flat.tolist()]


class CrossEncoderReranker(RerankerPort):
    """
    Scores (query, passage) pairs and reorders candidates.

    Default (SUNAT / Colab): ``cross-encoder/ms-marco-multilingual-MiniLM-L12-v2`` —
    multilingüe (incl. español), más ligero que BGE rerankers, adecuado para ~10–20
    candidatos y demos sin sobrecoste. Override con ``RAG_RERANKER_MODEL`` o el
    argumento ``model_name`` (p. ej. ``BAAI/bge-reranker-base`` si priorizas máxima precisión).
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        from src.config import get_settings

        s = get_settings()
        self._model_name = model_name or s.reranker_model_name
        self._batch_size = batch_size if batch_size is not None else s.reranker_batch_size
        self._model = self._load_model(self._model_name)

    def rerank(
        self,
        query: str,
        candidates: list[HybridChunkResult],
        top_k: int,
    ) -> list[RerankedChunkResult]:
        stripped = query.strip()
        if not stripped:
            raise ValueError("Query text cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not candidates:
            return []

        texts = [c.text.strip() or " " for c in candidates]
        pairs = [[stripped, t] for t in texts]
        raw = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        scores = _flatten_scores(raw)
        if len(scores) != len(candidates):
            raise ValueError(f"CrossEncoder returned {len(scores)} scores for {len(candidates)} candidates.")

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        limit = min(top_k, len(ranked))
        out: list[RerankedChunkResult] = []
        for pos, (cand, score) in enumerate(ranked[:limit], start=1):
            out.append(RerankedChunkResult.from_hybrid(cand, rerank_score=score, rerank_position=pos))
        logger.info("Reranked %s candidates -> top %s (model=%s).", len(candidates), limit, self._model_name)
        return out

    @staticmethod
    def _load_model(model_name: str) -> object:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for CrossEncoder reranking.") from exc
        logger.info("Loading cross-encoder: %s", model_name)
        return CrossEncoder(model_name)
