"""Lexical retrieval with BM25 over chunk texts."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re

from rank_bm25 import BM25Okapi

from src.domain.entities.chunk import Chunk

logger = logging.getLogger(__name__)

# Letters (incl. Spanish accents) and digits; keeps tokens like "5ta", "sunat"
_TOKEN_PATTERN = re.compile(r"[0-9]+|[\w]+", re.UNICODE)


def tokenize_spanish(text: str) -> list[str]:
    """Lowercase, Unicode-aware word tokens; simple heuristic for Spanish."""
    if not text or not text.strip():
        return []
    lowered = text.lower().strip()
    tokens = _TOKEN_PATTERN.findall(lowered)
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


@dataclass(slots=True)
class Bm25Hit:
    """One BM25 match with provenance fields."""

    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    text: str
    score: float


class BM25Retriever:
    """BM25Okapi index built from chunk texts."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def fit(self, chunks: list[Chunk]) -> None:
        """Tokenize chunk texts and fit BM25."""
        self._chunks = list(chunks)
        if not self._chunks:
            self._corpus_tokens = []
            self._bm25 = None
            logger.info("BM25 fit skipped: empty chunk list.")
            return

        self._corpus_tokens = []
        for chunk in self._chunks:
            toks = tokenize_spanish(chunk.text)
            self._corpus_tokens.append(toks if toks else ["__empty__"])

        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("BM25 index built: %s documents.", len(self._chunks))

    @property
    def is_fitted(self) -> bool:
        return self._bm25 is not None and bool(self._chunks)

    def search(self, query: str, top_k: int) -> list[Bm25Hit]:
        """Return top_k chunks by BM25 score for the query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not self.is_fitted or self._bm25 is None:
            logger.warning("BM25 search called on empty or unfitted index.")
            return []

        q_tokens = tokenize_spanish(query)
        if not q_tokens:
            q_tokens = ["__empty__"]

        raw_scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(
            range(len(raw_scores)),
            key=lambda i: raw_scores[i],
            reverse=True,
        )[:top_k]

        hits: list[Bm25Hit] = []
        for idx in ranked:
            chunk = self._chunks[idx]
            hits.append(
                Bm25Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    page=chunk.page,
                    text=chunk.text,
                    score=float(raw_scores[idx]),
                )
            )
        return hits
