from src.application.use_cases.answer_question import AnswerQuestionUseCase
from src.config import Settings, get_settings
from src.domain.entities.answer import Answer


class RAGPipeline:
    """Thin interface adapter for notebooks and scripts."""

    def __init__(
        self,
        answer_question_use_case: AnswerQuestionUseCase | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._answer_question_use_case = answer_question_use_case
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        """Expose loaded runtime settings."""
        return self._settings

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        """Execute QA flow through the configured use case."""
        if self._answer_question_use_case is None:
            raise RuntimeError("AnswerQuestionUseCase is not configured.")
        effective_top_k = top_k or self._settings.default_top_k
        return self._answer_question_use_case.execute(question=question, top_k=effective_top_k)
