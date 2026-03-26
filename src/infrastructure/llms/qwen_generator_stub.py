from src.application.ports.llm_port import LLMPort


class QwenGeneratorStub(LLMPort):
    """Stub Qwen generator adapter."""

    def generate(self, *, system: str | None = None, user: str) -> str:
        raise NotImplementedError("Answer generation is not implemented yet.")
