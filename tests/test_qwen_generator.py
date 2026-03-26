"""QwenGenerator con modelo y tokenizer mockeados (sin red ni GPU)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from src.infrastructure.llms.qwen_generator import QwenGenerator


class _FakeTokenizer:
    pad_token_id = 1
    eos_token_id = 2

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return "CHAT_PROMPT<<" + messages[-1]["content"][:20] + ">>"

    def __call__(self, prompt: str, return_tensors: str | None = None) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[10, 11, 12]])}

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "salida simulada del asistente"


class _FakeModel:
    def to(self, device: torch.device) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        extra = torch.tensor([[99]])
        return torch.cat([input_ids, extra], dim=1)


def _fake_load(self: QwenGenerator, model_name: str) -> tuple[object, object, torch.device]:
    return _FakeTokenizer(), _FakeModel(), torch.device("cpu")


@patch.object(QwenGenerator, "_load", _fake_load)
def test_qwen_generator_accepts_system_user_and_returns_str() -> None:
    gen = QwenGenerator(model_name="org/test-model")
    out = gen.generate(system="Eres útil.", user="Hola")
    assert isinstance(out, str)
    assert out == "salida simulada del asistente"


@patch.object(QwenGenerator, "_load", _fake_load)
def test_qwen_generator_model_name_configurable() -> None:
    gen = QwenGenerator(model_name="custom/Qwen-Stub")
    assert gen._model_name == "custom/Qwen-Stub"
    _ = gen.generate(user="x")
    # _load recibe el nombre configurado
    assert gen._model_name == "custom/Qwen-Stub"


@patch.object(QwenGenerator, "_load", _fake_load)
def test_qwen_generator_rejects_empty_user() -> None:
    gen = QwenGenerator(model_name="x")
    with pytest.raises(ValueError, match="empty"):
        gen.generate(user="   ")


@patch.object(QwenGenerator, "_load", _fake_load)
def test_qwen_generator_allows_no_system() -> None:
    gen = QwenGenerator(model_name="x")
    out = gen.generate(system=None, user="solo usuario")
    assert out == "salida simulada del asistente"
