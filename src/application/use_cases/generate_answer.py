"""Build SUNAT RAG prompt and call the LLM port."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.application.ports.llm_port import LLMPort
from src.application.use_cases.cite_answer import CiteAnswerUseCase
from src.domain.entities.answer import Answer, SentenceCitation
from src.domain.entities.retrieval import RerankedChunkResult

logger = logging.getLogger(__name__)

# Prompt de sistema: reglas globales (para slides: «instrucciones fijas del RAG»).
_SYSTEM_SUNAT_RAG = """Rol: asistente académico sobre tributación en Perú (SUNAT, impuestos, renta, declaraciones).

Reglas obligatorias:
1) Tu única fuente de hechos es el bloque «CONTEXTO DOCUMENTAL» del mensaje del usuario. No uses conocimiento externo.
2) No inventes artículos, porcentajes, plazos, requisitos ni procedimientos que no aparezcan literalmente o de forma clara en ese contexto.
3) Si el contexto no basta o la pregunta queda abierta: dilo en una frase al inicio; luego resume solo lo que sí está respaldado. Si nada aplica, indica que no hay evidencia suficiente en los fragmentos.
4) Respuesta en español, clara y concreta (párrafos cortos; viñetas solo si ordenan pasos o listas que el contexto permita)."""


def _format_context_fragment(idx: int, chunk: RerankedChunkResult) -> str:
    page = chunk.page if chunk.page is not None else "N/A"
    return (
        f"### [{idx}] Fuente: {chunk.source}\n"
        f"Página: {page} | ID fragmento: {chunk.chunk_id}\n\n"
        f"{chunk.text.strip()}\n"
    )


def build_sunat_rag_user_message(question: str, chunks: list[RerankedChunkResult]) -> str:
    """
    Construye el mensaje de usuario del RAG en tres partes explícitas:

    1. CONTEXTO DOCUMENTAL — solo fragmentos rerankeados.
    2. PREGUNTA DEL USUARIO — la consulta tal cual.
    3. TAREA — anclaje final (sin repetir reglas largas; el sistema ya las fija).

    Útil para notebooks, tests y para explicar el pipeline en defensa oral.
    """
    q = question.strip()
    lines: list[str] = [
        "--- CONTEXTO DOCUMENTAL ---",
        "La siguiente información es la única evidencia que puedes usar para responder.",
        "",
    ]
    if not chunks:
        lines.append("(No se recuperaron fragmentos. No inventes contenido normativo.)")
        lines.append("")
    else:
        for i, c in enumerate(chunks, start=1):
            lines.append(_format_context_fragment(i, c))
            lines.append("")

    lines.extend(
        [
            "--- PREGUNTA DEL USUARIO ---",
            q,
            "",
            "--- TAREA ---",
            "Responde la pregunta anterior usando únicamente el CONTEXTO DOCUMENTAL. "
            "Si algo no está respaldado por esos fragmentos, no lo afirmes.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(slots=True)
class GenerateAnswerUseCase:
    """Genera respuesta a partir de la pregunta y del top-k rerankeado."""

    llm: LLMPort
    citation_use_case: CiteAnswerUseCase | None = None

    def execute(self, question: str, reranked_chunks: list[RerankedChunkResult]) -> Answer:
        stripped_q = question.strip()
        if not stripped_q:
            raise ValueError("question cannot be empty.")

        user_msg = build_sunat_rag_user_message(stripped_q, reranked_chunks)
        answer_text = self.llm.generate(system=_SYSTEM_SUNAT_RAG, user=user_msg)

        context_meta = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source": c.source,
                "page": c.page,
                "rerank_position": c.rerank_position,
                "rerank_score": c.rerank_score,
                "text_preview": (c.text[:300] + "…") if len(c.text) > 300 else c.text,
            }
            for c in reranked_chunks
        ]

        citations: list[SentenceCitation] = []
        grounding_score: float | None = None
        hallucination_flag = False
        grounding_label: str | None = None

        if self.citation_use_case is not None:
            cite_result = self.citation_use_case.execute(answer_text, reranked_chunks)
            citations = cite_result.citations
            grounding_score = cite_result.grounding_score
            hallucination_flag = cite_result.hallucination_flag
            grounding_label = cite_result.grounding_label

        logger.info(
            "Answer generated for question len=%s using %s chunks (citations=%s grounding=%s).",
            len(stripped_q),
            len(reranked_chunks),
            len(citations),
            grounding_score,
        )

        return Answer(
            question=stripped_q,
            text=answer_text,
            citations=citations,
            grounding_score=grounding_score,
            hallucination_flag=hallucination_flag,
            metadata={
                "answer_text": answer_text,
                "prompt_system": _SYSTEM_SUNAT_RAG,
                "prompt_user": user_msg,
                "context_chunks": context_meta,
                "grounding_label": grounding_label,
                "sentence_citations": [
                    {
                        "sentence": c.sentence,
                        "chunk_id": c.chunk_id,
                        "source": c.source,
                        "page": c.page,
                        "similarity_score": c.similarity_score,
                    }
                    for c in citations
                ],
            },
        )
