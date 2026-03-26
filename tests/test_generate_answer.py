"""GenerateAnswerUseCase y build_sunat_rag_user_message (LLM mockeado)."""

from __future__ import annotations

import pytest

from src.application.ports.llm_port import LLMPort
from src.application.use_cases.generate_answer import (
    GenerateAnswerUseCase,
    build_sunat_rag_user_message,
)
from src.domain.entities.retrieval import RerankedChunkResult


def _sample_chunk(
    chunk_id: str = "c1",
    text: str = "Texto del chunk uno.",
    source: str = "https://sunat.gob.pe/x",
) -> RerankedChunkResult:
    return RerankedChunkResult(
        chunk_id=chunk_id,
        doc_id="d1",
        source=source,
        page=2,
        text=text,
        dense_score=0.5,
        sparse_score=1.0,
        hybrid_score=0.7,
        rerank_score=9.0,
        rerank_position=1,
    )


class _RecordingLLM(LLMPort):
    def __init__(self) -> None:
        self.last_system: str | None = None
        self.last_user: str | None = None

    def generate(self, *, system: str | None = None, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return "respuesta simulada"


class _FakeLLM(LLMPort):
    def generate(self, *, system: str | None = None, user: str) -> str:
        assert system and "SUNAT" in system and "CONTEXTO DOCUMENTAL" in system
        assert "--- CONTEXTO DOCUMENTAL ---" in user and "--- PREGUNTA DEL USUARIO ---" in user
        assert "### [1]" in user
        return "respuesta sintética"


def test_build_user_message_includes_question_and_context_fragments() -> None:
    chunks = [_sample_chunk(), _sample_chunk(chunk_id="c2", text="Segundo pasaje.")]
    msg = build_sunat_rag_user_message("  ¿Cuál es la tasa?  ", chunks)
    assert "--- CONTEXTO DOCUMENTAL ---" in msg
    assert "¿Cuál es la tasa?" in msg
    assert "### [1]" in msg and "### [2]" in msg
    assert "Texto del chunk uno." in msg and "Segundo pasaje." in msg
    assert "--- PREGUNTA DEL USUARIO ---" in msg
    assert "--- TAREA ---" in msg


def test_build_user_message_empty_context_signals_no_evidence_path() -> None:
    msg = build_sunat_rag_user_message("¿Algo?", [])
    assert "No se recuperaron fragmentos" in msg
    assert "No inventes contenido normativo" in msg
    assert "--- PREGUNTA DEL USUARIO ---" in msg
    assert "¿Algo?" in msg


def test_generate_answer_preserves_context_chunks_in_metadata() -> None:
    chunks = [_sample_chunk(), _sample_chunk(chunk_id="c2", text="otro")]
    llm = _RecordingLLM()
    uc = GenerateAnswerUseCase(llm=llm)
    ans = uc.execute("Pregunta?", chunks)

    assert ans.question == "Pregunta?"
    assert ans.text == "respuesta simulada"
    assert ans.metadata["answer_text"] == ans.text
    meta = ans.metadata["context_chunks"]
    assert len(meta) == 2
    assert meta[0]["chunk_id"] == "c1" and meta[1]["chunk_id"] == "c2"
    assert meta[0]["source"] == "https://sunat.gob.pe/x"
    assert meta[0]["rerank_position"] == 1
    assert "text_preview" in meta[0]


def test_generate_answer_prompt_built_from_question_and_context() -> None:
    chunks = [_sample_chunk()]
    llm = _RecordingLLM()
    GenerateAnswerUseCase(llm=llm).execute("¿Test?", chunks)
    assert llm.last_user is not None
    expected = build_sunat_rag_user_message("¿Test?", chunks)
    assert llm.last_user == expected
    assert llm.last_system is not None
    assert "evidencia suficiente" in llm.last_system.lower() or "evidencia" in llm.last_system


def test_generate_answer_empty_question_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        GenerateAnswerUseCase(llm=_RecordingLLM()).execute("   ", [_sample_chunk()])


def test_generate_answer_strips_question_whitespace() -> None:
    llm = _RecordingLLM()
    ans = GenerateAnswerUseCase(llm=llm).execute("  hola  ", [])
    assert ans.question == "hola"
    assert "hola" in llm.last_user


def test_generate_answer_empty_chunks_safe_and_empty_context_meta() -> None:
    llm = _RecordingLLM()
    ans = GenerateAnswerUseCase(llm=llm).execute("Solo pregunta", [])
    assert ans.metadata["context_chunks"] == []
    assert "No se recuperaron fragmentos" in llm.last_user


def test_generate_answer_system_instructs_insufficient_evidence() -> None:
    llm = _RecordingLLM()
    GenerateAnswerUseCase(llm=llm).execute("Q", [_sample_chunk()])
    assert llm.last_system is not None
    assert "no hay evidencia suficiente" in llm.last_system.lower()


def test_generate_answer_legacy_fake_llm_still_passes() -> None:
    ans = GenerateAnswerUseCase(llm=_FakeLLM()).execute("¿Pregunta de prueba?", [_sample_chunk()])
    assert ans.text == "respuesta sintética"
    assert ans.metadata["answer_text"] == ans.text
