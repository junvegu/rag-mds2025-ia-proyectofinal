from abc import ABC, abstractmethod

from src.domain.entities.answer import Answer
from src.domain.entities.retrieval import RetrievedChunk


class HallucinationGuardPort(ABC):
    """Estimate hallucination risk from answer and evidence."""

    @abstractmethod
    def score(self, answer: Answer, evidence: list[RetrievedChunk]) -> float:
        """Return risk score in [0.0, 1.0]."""
        raise NotImplementedError
