from typing import Any

from src.application.ports.evaluation_port import EvaluationPort
from src.domain.entities.answer import Answer


class MetricsEvaluatorStub(EvaluationPort):
    """Stub evaluator for ROUGE/BLEU-like metrics."""

    def evaluate(self, predictions: list[Answer], references: list[str]) -> dict[str, Any]:
        raise NotImplementedError("Evaluation is not implemented yet.")
