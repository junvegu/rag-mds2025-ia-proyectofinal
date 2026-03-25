"""Fuse dense (FAISS) and sparse (BM25) rankings with per-channel normalization and weighted sum."""

from __future__ import annotations

import logging
import math
from typing import Iterable

from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval import HybridChunkResult, HybridRetrieverConfig, RetrievedChunk
from src.infrastructure.retrieval.bm25_retriever import Bm25Hit

logger = logging.getLogger(__name__)


def _min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    """Map scores to [0, 1] per channel; flat lists become 1.0 for all (tie)."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {cid: 1.0 for cid in scores}
    return {cid: (scores[cid] - lo) / (hi - lo) for cid in scores}


def _softmax_norm(scores: dict[str, float], temperature: float) -> dict[str, float]:
    """
    Per-channel softmax over top-k candidates (stable log-sum-exp style).
    Comparable across BM25 vs FAISS raw scales within one query.
    """
    if not scores:
        return {}
    if temperature <= 0:
        raise ValueError("fusion_temperature must be positive.")
    ids = list(scores.keys())
    vals = [scores[i] for i in ids]
    m = max(vals)
    exps = [math.exp((v - m) / temperature) for v in vals]
    total = sum(exps)
    if total <= 0:
        n = len(ids)
        return {ids[i]: 1.0 / n for i in range(n)}
    return {ids[k]: exps[k] / total for k in range(len(ids))}


def _normalize_channel(
    scores: dict[str, float],
    mode: str,
    temperature: float,
) -> dict[str, float]:
    if mode == "softmax":
        return _softmax_norm(scores, temperature)
    if mode == "minmax":
        return _min_max_norm(scores)
    raise ValueError(f"Unknown score_normalization: {mode!r}")


def _chunk_lookup(dense_list: list[RetrievedChunk], sparse_list: list[Bm25Hit]) -> dict[str, Chunk]:
    """Prefer chunk objects from dense hits; fill gaps from sparse."""
    out: dict[str, Chunk] = {}
    for item in dense_list:
        out[item.chunk.chunk_id] = item.chunk
    for hit in sparse_list:
        if hit.chunk_id not in out:
            out[hit.chunk_id] = Chunk(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                text=hit.text,
                source=hit.source,
                page=hit.page,
            )
    return out


class HybridRetriever:
    """
    SUNAT-oriented hybrid: normalize dense and BM25 scores **por canal** (misma consulta),
    luego combinación convexa con pesos configurables.

    Por defecto: **softmax** + temperatura 1.0 (robusto a escalas distintas entre
    producto interno FAISS y BM25Okapi).
    """

    def __init__(self, config: HybridRetrieverConfig | None = None) -> None:
        self.config = config if config is not None else HybridRetrieverConfig()

    def fuse(self, dense_results: Iterable[RetrievedChunk], sparse_results: Iterable[Bm25Hit]) -> list[HybridChunkResult]:
        """Merge rankings; each chunk appears once; hybrid_score usa scores normalizados por canal."""
        dense_list = list(dense_results)
        sparse_list = list(sparse_results)
        cfg = self.config

        dense_by_id: dict[str, float] = {}
        for item in dense_list:
            if item.dense_score is not None:
                dense_by_id[item.chunk.chunk_id] = float(item.dense_score)

        sparse_by_id: dict[str, float] = {}
        for hit in sparse_list:
            sparse_by_id[hit.chunk_id] = float(hit.score)

        norm_d = _normalize_channel(dense_by_id, cfg.score_normalization, cfg.fusion_temperature)
        norm_s = _normalize_channel(sparse_by_id, cfg.score_normalization, cfg.fusion_temperature)

        all_ids = set(dense_by_id) | set(sparse_by_id)
        chunks = _chunk_lookup(dense_list, sparse_list)
        fused: list[HybridChunkResult] = []

        for cid in all_ids:
            d_raw = dense_by_id.get(cid)
            s_raw = sparse_by_id.get(cid)
            d_n = norm_d.get(cid, 0.0)
            s_n = norm_s.get(cid, 0.0)
            hybrid = cfg.dense_weight * d_n + cfg.sparse_weight * s_n
            ch = chunks[cid]

            fused.append(
                HybridChunkResult(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    source=ch.source,
                    page=ch.page,
                    text=ch.text,
                    dense_score=d_raw,
                    sparse_score=s_raw,
                    hybrid_score=hybrid,
                )
            )

        fused.sort(key=lambda r: r.hybrid_score, reverse=True)
        out = fused[: cfg.final_top_k]
        logger.info(
            "Hybrid fuse: norm=%s T=%s w_dense=%s w_sparse=%s dense=%s sparse=%s unique=%s final=%s",
            cfg.score_normalization,
            cfg.fusion_temperature,
            cfg.dense_weight,
            cfg.sparse_weight,
            len(dense_list),
            len(sparse_list),
            len(all_ids),
            len(out),
        )
        return out
