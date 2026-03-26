#!/usr/bin/env python3
"""
Ejemplo mínimo: citas por oración + grounding sin cargar Qwen.

Uso (desde la raíz del repo, con venv activo):

    python scripts/example_citation_grounding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.ports.llm_port import LLMPort
from src.application.use_cases.cite_answer import CiteAnswerUseCase
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.domain.entities.retrieval import RerankedChunkResult
from src.infrastructure.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings


class _EchoLLM(LLMPort):
    def generate(self, *, system: str | None = None, user: str) -> str:
        return (
            "La quinta categoría aplica a rentas laborales. "
            "El fraccionamiento ayuda a pagar deudas en cuotas."
        )


def main() -> int:
    chunks = [
        RerankedChunkResult(
            chunk_id="demo-1",
            doc_id="d1",
            source="https://www.gob.pe/ejemplo-tributario",
            page=None,
            text="La quinta categoría grava rentas de trabajo en relación de dependencia.",
            dense_score=0.9,
            sparse_score=1.0,
            hybrid_score=0.8,
            rerank_score=5.0,
            rerank_position=1,
        ),
        RerankedChunkResult(
            chunk_id="demo-2",
            doc_id="d2",
            source="https://renta.sunat.gob.pe/cartilla.pdf",
            page=12,
            text="El fraccionamiento permite pagar la deuda tributaria en cuotas.",
            dense_score=0.85,
            sparse_score=0.9,
            hybrid_score=0.75,
            rerank_score=4.0,
            rerank_position=2,
        ),
    ]

    embedder = SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-small", batch_size=8)
    cite_uc = CiteAnswerUseCase(embedder)
    gen_uc = GenerateAnswerUseCase(llm=_EchoLLM(), citation_use_case=cite_uc)

    answer = gen_uc.execute("¿Qué es quinta categoría y fraccionamiento?", chunks)

    print("=== answer_text ===")
    print(answer.text)
    print("\n=== grounding_score ===", answer.grounding_score)
    print("=== hallucination_flag ===", answer.hallucination_flag)
    print("=== grounding_label (metadata) ===", answer.metadata.get("grounding_label"))
    print("\n=== Citas por oración ===")
    for i, c in enumerate(answer.citations, start=1):
        print(f"{i}. sim={c.similarity_score:.4f} chunk={c.chunk_id} page={c.page}")
        print(f"   source={c.source[:70]}...")
        print(f"   sentence={c.sentence[:120]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
