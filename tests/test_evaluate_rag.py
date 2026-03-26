"""EvaluateRagUseCase y carga del JSON de evaluación."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.ports.llm_port import LLMPort
from src.application.use_cases.evaluate_rag import (
    EvaluateRagUseCase,
    academic_verdict,
    keyword_hit_ratio,
    unique_sources_from_chunks,
)
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.domain.entities.answer import Answer
from src.domain.entities.retrieval import HybridChunkResult, RerankedChunkResult
from src.infrastructure.evaluation.bleu_evaluator import BleuEvaluator
from src.infrastructure.evaluation.evaluation_dataset_loader import load_eval_questions
from src.infrastructure.evaluation.rouge_evaluator import RougeEvaluator


def _hybrid_one() -> HybridChunkResult:
    return HybridChunkResult(
        chunk_id="c1",
        doc_id="d1",
        source="https://x",
        page=1,
        text="texto de apoyo",
        dense_score=0.9,
        sparse_score=0.8,
        hybrid_score=0.85,
    )


def _reranked_one() -> RerankedChunkResult:
    h = _hybrid_one()
    return RerankedChunkResult(
        chunk_id=h.chunk_id,
        doc_id=h.doc_id,
        source=h.source,
        page=h.page,
        text=h.text,
        dense_score=h.dense_score,
        sparse_score=h.sparse_score,
        hybrid_score=h.hybrid_score,
        rerank_score=1.0,
        rerank_position=1,
    )


class _FakeRetrieve:
    def execute(self, query: str):
        return [_hybrid_one()]


class _FakeRerank:
    def execute(self, query: str, candidates, top_k: int):
        return [_reranked_one()]


class _FakeGen:
    def execute(self, question: str, reranked_chunks):
        return Answer(
            question=question,
            text="La quinta categoría grava rentas laborales en dependencia.",
            citations=[],
            grounding_score=0.8,
            hallucination_flag=False,
        )


def test_keyword_hit_ratio() -> None:
    assert keyword_hit_ratio("hola quinta categoría", ["quinta", "faltante"]) == 0.5
    assert keyword_hit_ratio("sin", []) is None


def test_load_eval_questions_from_repo_file() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/eval/eval_questions.json"
    items = load_eval_questions(path)
    assert len(items) >= 8
    assert all(i.question and i.reference_answer for i in items)


def test_evaluate_rag_aggregates() -> None:
    items = load_eval_questions(Path(__file__).resolve().parents[1] / "data/eval/eval_questions.json")[:2]
    uc = EvaluateRagUseCase(
        retrieve=_FakeRetrieve(),
        rerank=_FakeRerank(),
        generate=_FakeGen(),
        rouge=RougeEvaluator(),
        bleu=BleuEvaluator(),
        top_k=1,
    )
    report = uc.execute(items)
    assert report.total_questions == 2
    assert 0.0 <= report.mean_rouge1_f <= 1.0
    assert report.mean_grounding == pytest.approx(0.8)
    assert report.hallucination_count == 0
    ser = report.to_serializable()
    assert "global_metrics" in ser
    assert ser["global_metrics"]["total_questions"] == 2
    assert ser["global_metrics"]["total_hallucination_flags"] == 0
    assert "by_question" in ser and len(ser["by_question"]) == 2
    assert ser["by_question"][0]["sources_used"] == ["https://x"]
    assert "conclusion" in ser and ser["conclusion"]["verdict_key"] in (
        "strong_traceability",
        "solid",
        "acceptable",
    )
    assert "per_question_details" in ser


def test_unique_sources_from_chunks_order() -> None:
    a = _reranked_one()
    b = RerankedChunkResult(
        chunk_id="c2",
        doc_id="d2",
        source="https://y",
        page=1,
        text="otro",
        dense_score=0.5,
        sparse_score=0.5,
        hybrid_score=0.5,
        rerank_score=0.9,
        rerank_position=2,
    )
    c = RerankedChunkResult(
        chunk_id="c3",
        doc_id="d3",
        source="https://x",
        page=2,
        text="dup source",
        dense_score=0.4,
        sparse_score=0.4,
        hybrid_score=0.4,
        rerank_score=0.8,
        rerank_position=3,
    )
    assert unique_sources_from_chunks([a, b, c]) == ["https://x", "https://y"]


def test_academic_verdict_strong() -> None:
    out = academic_verdict(
        avg_rouge_1=0.3,
        avg_rouge_l=0.25,
        avg_grounding=0.7,
        hallucination_rate=0.1,
        total_questions=10,
    )
    assert out["verdict_key"] == "strong_traceability"


def test_academic_verdict_no_grounding() -> None:
    out = academic_verdict(
        avg_rouge_1=0.25,
        avg_rouge_l=0.26,
        avg_grounding=None,
        hallucination_rate=0.2,
        total_questions=5,
    )
    assert out["verdict_key"] == "solid"


class _StubLLM(LLMPort):
    def generate(self, *, system: str | None = None, user: str) -> str:
        return "stub"


def test_evaluate_rag_empty_set_raises() -> None:
    uc = EvaluateRagUseCase(
        retrieve=_FakeRetrieve(),
        rerank=_FakeRerank(),
        generate=GenerateAnswerUseCase(llm=_StubLLM()),
        rouge=RougeEvaluator(),
        bleu=BleuEvaluator(),
    )
    with pytest.raises(ValueError, match="vacío"):
        uc.execute([])
