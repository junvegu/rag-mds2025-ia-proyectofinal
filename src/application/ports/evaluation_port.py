from abc import ABC, abstractmethod
from typing import Any

from src.domain.entities.answer import Answer


class EvaluationPort(ABC):
    """Evaluate generated answers using automatic metrics."""

    @abstractmethod
    def evaluate(self, predictions: list[Answer], references: list[str]) -> dict[str, Any]:
        """Return metric scores (e.g., ROUGE/BLEU)."""
        raise NotImplementedError
