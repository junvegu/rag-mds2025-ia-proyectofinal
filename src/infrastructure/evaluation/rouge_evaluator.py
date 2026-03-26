"""ROUGE-1 y ROUGE-L (F1) entre hipótesis y referencia.

ROUGE mide solapamiento léxico; en español sin stemmer es una aproximación útil
pero no sustituye juicio experto ni métricas de factualidad.
"""

from __future__ import annotations

from dataclasses import dataclass

from rouge_score import rouge_scorer


@dataclass(slots=True)
class RougeScores:
    rouge1_f: float
    rouge1_p: float
    rouge1_r: float
    rougeL_f: float
    rougeL_p: float
    rougeL_r: float


class RougeEvaluator:
    """Compara una hipótesis (respuesta generada) contra una referencia."""

    def __init__(self, *, use_stemmer: bool = False) -> None:
        # Stemmer en inglés; para español suele ser más seguro desactivarlo.
        self._scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=use_stemmer)

    def score_pair(self, hypothesis: str, reference: str) -> RougeScores:
        """``reference`` = respuesta gold; ``hypothesis`` = salida del modelo."""
        h = (hypothesis or "").strip()
        r = (reference or "").strip()
        if not h and not r:
            return RougeScores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        if not h or not r:
            return RougeScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s = self._scorer.score(r, h)
        return RougeScores(
            rouge1_f=float(s["rouge1"].fmeasure),
            rouge1_p=float(s["rouge1"].precision),
            rouge1_r=float(s["rouge1"].recall),
            rougeL_f=float(s["rougeL"].fmeasure),
            rougeL_p=float(s["rougeL"].precision),
            rougeL_r=float(s["rougeL"].recall),
        )
