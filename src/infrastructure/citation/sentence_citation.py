"""Segmentación de respuesta y alineación oración ↔ chunk por similitud coseno."""

from __future__ import annotations

import logging
import re
import numpy as np

from src.domain.entities.answer import SentenceCitation
from src.domain.entities.retrieval import RerankedChunkResult

logger = logging.getLogger(__name__)

# Fin de oración en español (conservador; evita depender de NLTK).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_answer_sentences(answer_text: str) -> list[str]:
    """Divide la respuesta en oraciones no vacías (heurística ligera)."""
    if not answer_text or not answer_text.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(answer_text.strip())
    out: list[str] = []
    for p in parts:
        s = p.strip()
        if len(s) < 2:
            continue
        out.append(s)
    return out


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def cosine_similarity_matrix(sentence_vectors: np.ndarray, chunk_vectors: np.ndarray) -> np.ndarray:
    """
    Matriz (n_sentences, n_chunks) de similitud coseno.
    Vectores fila; se normalizan en L2 antes del producto interno.
    """
    if sentence_vectors.size == 0 or chunk_vectors.size == 0:
        return np.zeros((sentence_vectors.shape[0], chunk_vectors.shape[0]), dtype=np.float64)
    s = _l2_normalize_rows(sentence_vectors.astype(np.float64, copy=False))
    c = _l2_normalize_rows(chunk_vectors.astype(np.float64, copy=False))
    return s @ c.T


def assign_best_chunk_per_sentence(
    sentences: list[str],
    sentence_embeddings: list[list[float]],
    chunks: list[RerankedChunkResult],
    chunk_embeddings: list[list[float]],
) -> list[SentenceCitation]:
    """
    Para cada oración, elige el chunk con mayor similitud coseno.

    ``sentence_embeddings`` y ``chunk_embeddings`` deben alinearse con ``sentences`` y ``chunks``.
    """
    if not sentences:
        return []
    if not chunks or not chunk_embeddings:
        logger.warning("assign_best_chunk_per_sentence: sin chunks; no hay evidencia que asignar.")
        return []

    s_mat = np.asarray(sentence_embeddings, dtype=np.float64)
    c_mat = np.asarray(chunk_embeddings, dtype=np.float64)
    if s_mat.shape[0] != len(sentences):
        raise ValueError("sentence_embeddings length must match sentences.")
    if c_mat.shape[0] != len(chunks):
        raise ValueError("chunk_embeddings length must match chunks.")

    sims = cosine_similarity_matrix(s_mat, c_mat)
    best_idx = np.argmax(sims, axis=1)
    citations: list[SentenceCitation] = []
    for i, sent in enumerate(sentences):
        j = int(best_idx[i])
        score = float(sims[i, j])
        ch = chunks[j]
        citations.append(
            SentenceCitation(
                sentence=sent,
                chunk_id=ch.chunk_id,
                source=ch.source,
                page=ch.page,
                similarity_score=score,
            )
        )
    return citations


def grounding_label_from_score(score: float) -> str:
    """Etiqueta legible según umbrales del proyecto."""
    if score >= 0.75:
        return "well_grounded"
    if score >= 0.55:
        return "partially_grounded"
    return "possible_hallucination"


def hallucination_flag_from_grounding(score: float) -> bool:
    """True si el grounding medio sugiere posible alucinación (< 0.55)."""
    return score < 0.55
