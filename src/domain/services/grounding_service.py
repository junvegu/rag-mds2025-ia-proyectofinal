from src.domain.entities.answer import Answer


class GroundingService:
    """Utility service for grounding validation and scoring."""

    def is_grounded(self, answer: Answer) -> bool:
        """Return True when at least one citation is present."""
        return len(answer.citations) > 0

    def estimate_grounding_score(self, answer: Answer) -> float:
        """Estimate grounding score from sentence-level citation coverage."""
        sentence_ids = {c.sentence_index for c in answer.citations if c.sentence_index is not None}
        if not sentence_ids:
            return 0.0
        total_sentences = int(answer.metadata.get("sentence_count", len(sentence_ids)))
        # Minimal heuristic until full evaluator is implemented.
        return min(1.0, len(sentence_ids) / max(1, total_sentences))
