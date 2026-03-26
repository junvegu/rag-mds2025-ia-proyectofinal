from src.domain.entities.answer import Answer


class GroundingService:
    """Utilidades sobre ``Answer`` ya enriquecido con citas y ``grounding_score``."""

    def is_grounded(self, answer: Answer) -> bool:
        """True si hay score agregado >= 0.55 o, en su defecto, al menos una cita."""
        if answer.grounding_score is not None:
            return answer.grounding_score >= 0.55
        return len(answer.citations) > 0

    def estimate_grounding_score(self, answer: Answer) -> float:
        """Prefiere ``answer.grounding_score``; si no, media de ``similarity_score`` por oración."""
        if answer.grounding_score is not None:
            return answer.grounding_score
        if not answer.citations:
            return 0.0
        return sum(c.similarity_score for c in answer.citations) / len(answer.citations)
