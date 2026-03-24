from src.application.ports.hallucination_guard_port import HallucinationGuardPort
from src.domain.entities.answer import Answer
from src.domain.entities.retrieval import RetrievedChunk


class HallucinationGuardStub(HallucinationGuardPort):
    """Stub hallucination guard adapter."""

    def score(self, answer: Answer, evidence: list[RetrievedChunk]) -> float:
        raise NotImplementedError("Hallucination guard is not implemented yet.")
