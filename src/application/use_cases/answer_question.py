from dataclasses import dataclass

from src.application.ports.hallucination_guard_port import HallucinationGuardPort
from src.application.ports.llm_port import LLMPort
from src.application.ports.reranker_port import RerankerPort
from src.application.ports.retriever_port import RetrieverPort
from src.domain.entities.answer import Answer


@dataclass(slots=True)
class AnswerQuestionUseCase:
    """Application use case for question-answering with RAG."""

    retriever: RetrieverPort
    reranker: RerankerPort
    generator: LLMPort
    hallucination_guard: HallucinationGuardPort | None = None

    def execute(self, question: str, top_k: int = 5) -> Answer:
        """Return a placeholder response while the RAG logic is pending."""
        return Answer(
            question=question,
            text="RAG pipeline bootstrap ready. Real retrieval/generation is pending.",
            metadata={"status": "stub", "top_k": top_k},
        )
