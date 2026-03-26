"""Qwen (transformers) generator for RAG answers."""

from __future__ import annotations

import logging

import torch

from src.application.ports.llm_port import LLMPort
from src.infrastructure.huggingface_auth_fallback import (
    huggingface_invalid_env_token_error,
    huggingface_public_hub_session,
)

logger = logging.getLogger(__name__)


class QwenGenerator(LLMPort):
    """
    Carga un modelo causal instruct (por defecto Qwen2.5 pequeño para Colab).

    Colab T4: ``Qwen/Qwen2.5-1.5B-Instruct`` suele ser viable en fp16; subir a
    ``Qwen2.5-3B-Instruct`` si hay más VRAM.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> None:
        from src.config import get_settings

        s = get_settings()
        self._model_name = model_name or s.generation_model_name
        self._max_new_tokens = max_new_tokens if max_new_tokens is not None else s.generation_max_new_tokens
        self._temperature = temperature if temperature is not None else s.generation_temperature
        self._top_p = top_p if top_p is not None else s.generation_top_p
        self._do_sample = do_sample if do_sample is not None else s.generation_do_sample

        self._tokenizer, self._model, self._device = self._load(self._model_name)

    def generate(self, *, system: str | None = None, user: str) -> str:
        if not user.strip():
            raise ValueError("user message cannot be empty.")

        messages: list[dict[str, str]] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": user.strip()})

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        raw = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(raw, "to"):
            inputs = raw.to(self._device)
        else:
            inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self._do_sample:
            gen_kwargs["temperature"] = self._temperature
            gen_kwargs["top_p"] = self._top_p

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0, input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logger.info("Generated %s chars (model=%s).", len(text), self._model_name)
        return text

    def _load(self, model_name: str) -> tuple[object, object, torch.device]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        logger.info("Loading Qwen tokenizer/model: %s (device=%s, dtype=%s)", model_name, device, dtype)

        def _load_pair(*, use_token: bool | None) -> tuple[object, object]:
            tok_kw: dict = {"trust_remote_code": True}
            mod_kw: dict = {"torch_dtype": dtype, "trust_remote_code": True}
            if use_token is False:
                tok_kw["token"] = False
                mod_kw["token"] = False
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kw)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(model_name, **mod_kw)
            return tokenizer, model

        try:
            tokenizer, model = _load_pair(use_token=None)
        except Exception as exc:  # noqa: BLE001
            if not huggingface_invalid_env_token_error(exc):
                raise
            logger.warning(
                "HF Hub rejected auth for %s; retrying with token=False (fix or unset HF_TOKEN).",
                model_name,
            )
            with huggingface_public_hub_session():
                tokenizer, model = _load_pair(use_token=False)
        model.to(device)
        model.eval()
        return tokenizer, model, device
