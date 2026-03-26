from abc import ABC, abstractmethod


class LLMPort(ABC):
    """Causal LM generation from system + user strings (chat-style)."""

    @abstractmethod
    def generate(self, *, system: str | None = None, user: str) -> str:
        """Return assistant text only (no chat template wrapper)."""
        raise NotImplementedError
