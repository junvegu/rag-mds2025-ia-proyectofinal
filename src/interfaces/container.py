from src.config import Settings
from src.application.use_cases.answer_question import AnswerQuestionUseCase
from src.infrastructure.evaluation import HallucinationGuardStub
from src.infrastructure.llms import QwenGeneratorStub
from src.infrastructure.rerankers import CrossEncoderRerankerStub
from src.infrastructure.retrieval import HybridRetrieverStub
from src.interfaces.rag_pipeline import RAGPipeline


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    """Build pipeline with explicit stub wiring for local development."""
    use_case = AnswerQuestionUseCase(
        retriever=HybridRetrieverStub(),
        reranker=CrossEncoderRerankerStub(),
        generator=QwenGeneratorStub(),
        hallucination_guard=HallucinationGuardStub(),
    )
    return RAGPipeline(answer_question_use_case=use_case, settings=settings)
