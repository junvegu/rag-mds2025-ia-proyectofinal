"""Post-generación: citas por oración y grounding agregado vía embeddings existentes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.application.ports.embedding_port import EmbeddingPort
from src.domain.entities.answer import SentenceCitation
from src.domain.entities.retrieval import RerankedChunkResult
from src.infrastructure.citation.sentence_citation import (
    assign_best_chunk_per_sentence,
    grounding_label_from_score,
    hallucination_flag_from_grounding,
    split_answer_sentences,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CitationGroundingResult:
    """Salida del alineamiento oración ↔ chunk."""

    citations: list[SentenceCitation]
    grounding_score: float
    grounding_label: str
    hallucination_flag: bool


class CiteAnswerUseCase:
    """
    Una sola llamada a ``embed_texts`` para chunks + oraciones (sin re-embedder chunks en otro sitio).

    Reutiliza el mismo adaptador de embeddings que el índice denso (p. ej. e5-small normalizado).
    """

    def __init__(self, embedding_adapter: EmbeddingPort) -> None:
        self._embed = embedding_adapter

    def execute(self, answer_text: str, context_chunks: list[RerankedChunkResult]) -> CitationGroundingResult:
        sentences = split_answer_sentences(answer_text)
        if not sentences:
            return CitationGroundingResult(
                citations=[],
                grounding_score=1.0,
                grounding_label=grounding_label_from_score(1.0),
                hallucination_flag=False,
            )

        if not context_chunks:
            logger.info("CiteAnswer: contexto vacío; grounding mínimo.")
            return CitationGroundingResult(
                citations=[],
                grounding_score=0.0,
                grounding_label=grounding_label_from_score(0.0),
                hallucination_flag=True,
            )

        chunk_texts = [c.text.strip() or " " for c in context_chunks]
        # Un solo batch: primero chunks (orden estable), luego oraciones.
        batch = chunk_texts + sentences
        vectors = self._embed.embed_texts(batch)
        if len(vectors) != len(batch):
            raise ValueError("embedding adapter returned unexpected vector count.")

        n_c = len(context_chunks)
        chunk_embs = vectors[:n_c]
        sent_embs = vectors[n_c:]

        citations = assign_best_chunk_per_sentence(
            sentences=sentences,
            sentence_embeddings=sent_embs,
            chunks=context_chunks,
            chunk_embeddings=chunk_embs,
        )
        scores = [c.similarity_score for c in citations]
        grounding = float(sum(scores) / len(scores)) if scores else 0.0
        label = grounding_label_from_score(grounding)
        hallu = hallucination_flag_from_grounding(grounding)

        logger.info(
            "Citations: %s sentences -> %s chunks, grounding_mean=%.4f label=%s hallucination_flag=%s",
            len(sentences),
            len(context_chunks),
            grounding,
            label,
            hallu,
        )

        return CitationGroundingResult(
            citations=citations,
            grounding_score=grounding,
            grounding_label=label,
            hallucination_flag=hallu,
        )
