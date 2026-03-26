"""Carga del JSON de evaluación en ``data/eval/``."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvalQuestionItem:
    id: str
    question: str
    reference_answer: str
    expected_keywords: list[str]
    topic: str | None


def load_eval_questions(path: str | Path) -> list[EvalQuestionItem]:
    """Lee ``eval_questions.json`` con clave ``items``."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    items = raw.get("items")
    if not isinstance(items, list):
        raise ValueError("JSON de evaluación debe contener una lista 'items'.")
    out: list[EvalQuestionItem] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("id", "")).strip() or f"row_{len(out)}"
        q = str(row.get("question", "")).strip()
        ref = str(row.get("reference_answer", "")).strip()
        kws = row.get("expected_keywords") or []
        if not isinstance(kws, list):
            kws = []
        kws = [str(x).strip() for x in kws if str(x).strip()]
        topic = row.get("topic")
        topic_s = str(topic).strip() if topic else None
        out.append(
            EvalQuestionItem(
                id=qid,
                question=q,
                reference_answer=ref,
                expected_keywords=kws,
                topic=topic_s,
            )
        )
    logger.info("Cargadas %s preguntas de evaluación desde %s", len(out), p)
    return out
