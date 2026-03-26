"""Evalúa el RAG end-to-end: pipeline + ROUGE/BLEU + grounding por pregunta."""

from __future__ import annotations

import logging
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.rerank_context import RerankContextUseCase
from src.application.use_cases.retrieve_context import RetrieveContextUseCase
from src.domain.entities.retrieval import RerankedChunkResult
from src.infrastructure.evaluation.evaluation_dataset_loader import EvalQuestionItem
from src.infrastructure.evaluation.bleu_evaluator import BleuEvaluator
from src.infrastructure.evaluation.rouge_evaluator import RougeEvaluator

logger = logging.getLogger(__name__)

VerdictKey = Literal["strong_traceability", "solid", "acceptable"]


def _normalize_for_keyword(text: str) -> str:
    lowered = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )


def keyword_hit_ratio(answer: str, keywords: list[str]) -> float | None:
    """Fracción de palabras clave presentes en la respuesta (aprox. léxica)."""
    if not keywords:
        return None
    an = _normalize_for_keyword(answer)
    hits = sum(1 for k in keywords if _normalize_for_keyword(k) in an)
    return hits / len(keywords)


def unique_sources_from_chunks(chunks: list[RerankedChunkResult]) -> list[str]:
    """Fuentes únicas en orden de aparición (contexto rerankeado)."""
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        s = (c.source or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def academic_verdict(
    *,
    avg_rouge_1: float,
    avg_rouge_l: float,
    avg_grounding: float | None,
    hallucination_rate: float,
    total_questions: int,
) -> dict[str, Any]:
    """
    Conclusión automática en tres bandas (reglas fijas y auditables).

    **Fuerte:** buen anclaje semántico al contexto + ROUGE-L razonable + pocas banderas.
    **Sólido:** equilibrio medio en grounding y overlap léxico + tolerancia moderada a flags.
    **Aceptable:** cumple el cierre del proyecto; mejora continua posible.

    Si no hay grounding (None), solo se usan ROUGE y tasa de flags (se indica en ``checks``).
    """
    checks: list[dict[str, Any]] = []
    g_ok_strong = False
    g_ok_solid = False

    if avg_grounding is not None:
        checks.append(
            {
                "metric": "avg_grounding_score",
                "value": round(avg_grounding, 4),
                "note": "Media de similitud oración–chunk (embeddings del mismo modelo que retrieval).",
            }
        )
        g_ok_strong = avg_grounding >= 0.68
        g_ok_solid = avg_grounding >= 0.55
        checks.append(
            {
                "rule": "grounding_strong",
                "threshold": ">= 0.68",
                "passed": g_ok_strong,
            }
        )
        checks.append(
            {
                "rule": "grounding_solid",
                "threshold": ">= 0.55",
                "passed": g_ok_solid,
            }
        )
    else:
        checks.append(
            {
                "metric": "avg_grounding_score",
                "value": None,
                "note": "No disponible (sin citas/grounding en esta corrida).",
            }
        )

    rl_strong = avg_rouge_l >= 0.22
    rl_solid = avg_rouge_l >= 0.14
    r1_solid = avg_rouge_1 >= 0.18
    flags_strong = hallucination_rate <= 0.25
    flags_solid = hallucination_rate <= 0.45

    checks.extend(
        [
            {"rule": "rouge_l_strong", "threshold": ">= 0.22", "passed": rl_strong},
            {"rule": "rouge_l_solid", "threshold": ">= 0.14", "passed": rl_solid},
            {"rule": "rouge_1_solid_fallback", "threshold": ">= 0.18", "passed": r1_solid},
            {
                "rule": "hallucination_rate_strong",
                "threshold": "<= 25% preguntas con flag",
                "passed": flags_strong,
                "value": round(hallucination_rate, 4),
            },
            {
                "rule": "hallucination_rate_solid",
                "threshold": "<= 45% preguntas con flag",
                "passed": flags_solid,
                "value": round(hallucination_rate, 4),
            },
        ]
    )

    rules_text = [
        "Nivel fuerte: grounding medio ≥ 0.68, ROUGE-L medio ≥ 0.22, y ≤25% de preguntas con hallucination_flag.",
        "Nivel sólido: grounding medio ≥ 0.55, (ROUGE-L ≥ 0.14 o ROUGE-1 ≥ 0.18), y ≤45% de flags.",
        "Nivel aceptable: cualquier otro caso (sigue siendo informe válido para cierre; revisar manualmente).",
        "ROUGE no mide veracidad legal; solo solapamiento léxico con la referencia escrita.",
    ]

    verdict: VerdictKey
    summary_es: str

    if avg_grounding is not None:
        if g_ok_strong and rl_strong and flags_strong:
            verdict = "strong_traceability"
            summary_es = (
                "Sistema fuerte con buena trazabilidad: grounding alto, overlap léxico adecuado "
                "y pocas alertas de alucinación automática."
            )
        elif g_ok_solid and (rl_solid or r1_solid) and flags_solid:
            verdict = "solid"
            summary_es = (
                "Sistema sólido: anclaje y métricas léxicas en rango intermedio defendible para trabajo académico."
            )
        else:
            verdict = "acceptable"
            summary_es = (
                "Sistema aceptable: resultados útiles para demostración y análisis; conviene revisión humana "
                "y posible ajuste de corpus o prompts."
            )
    else:
        if avg_rouge_l >= 0.24 and avg_rouge_1 >= 0.20 and hallucination_rate <= 0.30:
            verdict = "solid"
            summary_es = "Sistema sólido (sin media de grounding): basado en ROUGE y tasa de flags."
        elif avg_rouge_1 >= 0.12 or avg_rouge_l >= 0.10:
            verdict = "acceptable"
            summary_es = "Sistema aceptable (sin media de grounding): métricas léxicas moderadas."
        else:
            verdict = "acceptable"
            summary_es = "Sistema aceptable: métricas bajas o contexto limitado; priorizar revisión cualitativa."

    return {
        "verdict_key": verdict,
        "verdict_es": {
            "strong_traceability": "Sistema fuerte con buena trazabilidad",
            "solid": "Sistema sólido",
            "acceptable": "Sistema aceptable",
        }[verdict],
        "summary_es": summary_es,
        "rules_stated": rules_text,
        "checks": checks,
        "total_questions": total_questions,
    }


@dataclass(slots=True)
class PerQuestionEvalResult:
    question_id: str
    topic: str | None
    question: str
    reference_answer: str
    generated_answer: str
    total_seconds: float
    rouge1_f: float
    rougeL_f: float
    bleu: float
    grounding_score: float | None
    hallucination_flag: bool
    answer_char_count: int
    keyword_hit_ratio: float | None
    sources_used: list[str]


@dataclass(slots=True)
class RagEvaluationReport:
    """Resultados por ítem y agregados (útiles para JSON, README y slides)."""

    per_question: list[PerQuestionEvalResult]
    mean_rouge1_f: float
    mean_rougeL_f: float
    mean_bleu: float
    mean_grounding: float | None
    hallucination_count: int
    total_questions: int
    disclaimer: str = field(
        default=(
            "ROUGE/BLEU son métricas léxicas aproximadas; no miden factualidad legal. "
            "Combinar con grounding_score y revisión humana."
        )
    )

    def to_serializable(self) -> dict[str, Any]:
        """Estructura plana para notebook, slides y ``evaluation_report.json``."""
        n = max(1, self.total_questions)
        hallu_rate = self.hallucination_count / n

        global_metrics = {
            "avg_rouge_1": round(self.mean_rouge1_f, 6),
            "avg_rouge_l": round(self.mean_rougeL_f, 6),
            "avg_grounding_score": (round(self.mean_grounding, 6) if self.mean_grounding is not None else None),
            "avg_bleu": round(self.mean_bleu, 4),
            "total_questions": self.total_questions,
            "total_hallucination_flags": self.hallucination_count,
            "hallucination_rate": round(hallu_rate, 6),
        }

        by_question: list[dict[str, Any]] = []
        for p in self.per_question:
            by_question.append(
                {
                    "question_id": p.question_id,
                    "topic": p.topic,
                    "question": p.question,
                    "reference_answer": p.reference_answer,
                    "generated_answer": p.generated_answer,
                    "rouge_1": round(p.rouge1_f, 6),
                    "rouge_l": round(p.rougeL_f, 6),
                    "bleu": round(p.bleu, 4),
                    "grounding_score": (round(p.grounding_score, 6) if p.grounding_score is not None else None),
                    "hallucination_flag": p.hallucination_flag,
                    "sources_used": list(p.sources_used),
                    "time_seconds": round(p.total_seconds, 4),
                    "keyword_hit_ratio": p.keyword_hit_ratio,
                }
            )

        conclusion = academic_verdict(
            avg_rouge_1=self.mean_rouge1_f,
            avg_rouge_l=self.mean_rougeL_f,
            avg_grounding=self.mean_grounding,
            hallucination_rate=hallu_rate,
            total_questions=self.total_questions,
        )

        return {
            "disclaimer": self.disclaimer,
            "global_metrics": global_metrics,
            "conclusion": conclusion,
            "by_question": by_question,
            "per_question_details": [asdict(p) for p in self.per_question],
        }


@dataclass(slots=True)
class EvaluateRagUseCase:
    """
    Ejecuta recuperación → rerank → generación (con citas si ``generate`` las incluye)
    y puntúa frente a referencias manuales.
    """

    retrieve: RetrieveContextUseCase
    rerank: RerankContextUseCase
    generate: GenerateAnswerUseCase
    rouge: RougeEvaluator
    bleu: BleuEvaluator
    top_k: int = 5

    def execute(self, items: list[EvalQuestionItem]) -> RagEvaluationReport:
        if not items:
            raise ValueError("evaluation set vacío.")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0.")

        per: list[PerQuestionEvalResult] = []
        rouge1_sum = 0.0
        rougeL_sum = 0.0
        bleu_sum = 0.0
        ground_vals: list[float] = []
        hallu_count = 0

        for it in items:
            t0 = time.perf_counter()
            hybrid = self.retrieve.execute(it.question)
            reranked = self.rerank.execute(it.question, hybrid, self.top_k)
            answer = self.generate.execute(it.question, reranked)
            elapsed = time.perf_counter() - t0
            sources_used = unique_sources_from_chunks(reranked)

            rs = self.rouge.score_pair(answer.text, it.reference_answer)
            bl = self.bleu.score_pair(answer.text, it.reference_answer)
            kw = keyword_hit_ratio(answer.text, it.expected_keywords)
            g = answer.grounding_score
            if g is not None:
                ground_vals.append(g)

            if answer.hallucination_flag:
                hallu_count += 1

            rouge1_sum += rs.rouge1_f
            rougeL_sum += rs.rougeL_f
            bleu_sum += bl

            per.append(
                PerQuestionEvalResult(
                    question_id=it.id,
                    topic=it.topic,
                    question=it.question,
                    reference_answer=it.reference_answer,
                    generated_answer=answer.text,
                    total_seconds=elapsed,
                    rouge1_f=rs.rouge1_f,
                    rougeL_f=rs.rougeL_f,
                    bleu=bl,
                    grounding_score=g,
                    hallucination_flag=answer.hallucination_flag,
                    answer_char_count=len(answer.text.strip()),
                    keyword_hit_ratio=kw,
                    sources_used=sources_used,
                )
            )
            logger.info(
                "Eval %s: rouge1_f=%.3f rougeL_f=%.3f bleu=%.1f grounding=%s hallu=%s (%.2fs) sources=%s",
                it.id,
                rs.rouge1_f,
                rs.rougeL_f,
                bl,
                g,
                answer.hallucination_flag,
                elapsed,
                len(sources_used),
            )

        n = len(per)
        mean_g = sum(ground_vals) / len(ground_vals) if ground_vals else None

        return RagEvaluationReport(
            per_question=per,
            mean_rouge1_f=rouge1_sum / n,
            mean_rougeL_f=rougeL_sum / n,
            mean_bleu=bleu_sum / n,
            mean_grounding=mean_g,
            hallucination_count=hallu_count,
            total_questions=n,
        )
