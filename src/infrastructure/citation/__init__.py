"""Citation and grounding helpers (post-hoc sentence → chunk alignment)."""

from .sentence_citation import (
    assign_best_chunk_per_sentence,
    grounding_label_from_score,
    split_answer_sentences,
)

__all__ = [
    "assign_best_chunk_per_sentence",
    "grounding_label_from_score",
    "split_answer_sentences",
]
