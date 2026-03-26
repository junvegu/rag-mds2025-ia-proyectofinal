"""BLEU a nivel de oración con sacrebleu (complemento opcional a ROUGE).

BLEU favorece coincidencias de n-gramas; para respuestas libres en español
suele ser estricto y debe interpretarse junto con ROUGE y grounding.
"""

from __future__ import annotations

import sacrebleu


class BleuEvaluator:
    def score_pair(self, hypothesis: str, reference: str) -> float:
        """Retorna BLEU en escala 0–100 (convención sacrebleu)."""
        h = (hypothesis or "").strip()
        r = (reference or "").strip()
        if not h and not r:
            return 100.0
        if not h or not r:
            return 0.0
        out = sacrebleu.sentence_bleu(h, [r])
        return float(out.score)
