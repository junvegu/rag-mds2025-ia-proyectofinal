from src.application.ports.llm_port import LLMPort
from src.domain.entities.answer import Answer
from src.domain.entities.retrieval import RetrievedChunk


class QwenGeneratorStub(LLMPort):
    """Stub Qwen generator adapter."""

    def generate(self, question: str, context: list[RetrievedChunk]) -> Answer:
        raise NotImplementedError("Answer generation is not implemented yet.")
