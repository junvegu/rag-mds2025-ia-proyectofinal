from abc import ABC, abstractmethod

from src.domain.entities.answer import Answer
from src.domain.entities.retrieval import RetrievedChunk


class LLMPort(ABC):
    """Generate grounded answers from ranked context."""

    @abstractmethod
    def generate(self, question: str, context: list[RetrievedChunk]) -> Answer:
        """Generate an answer object with optional citations."""
        raise NotImplementedError
