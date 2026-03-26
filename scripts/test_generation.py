"""Validación end-to-end SUNAT: pipeline hasta respuesta con Qwen; JSON opcional.

Si ves ``segmentation fault`` justo después de cargar FAISS: en **macOS** suele ser
conflicto **OpenMP** entre PyTorch (MPS) y faiss-cpu; los scripts llaman
``apply_darwin_openmp_mitigations()`` (o exporta ``OMP_NUM_THREADS=1`` y
``KMP_DUPLICATE_LIB_OK=TRUE``). Con **Python 3.13+** también hay riesgo por ruedas
nativas; usa **3.11/3.12**. El aviso de ``loky``/semáforos es consecuencia del crash.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import unicodedata
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime_env import apply_darwin_openmp_mitigations

apply_darwin_openmp_mitigations()

from src.application.use_cases.build_bm25_index import BuildBm25IndexUseCase
from src.application.use_cases.build_vector_index import BuildVectorIndexUseCase
from src.application.use_cases.chunk_documents import ChunkDocumentsUseCase
from src.application.use_cases.embed_chunks import EmbedChunksUseCase
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.load_documents import LoadDocumentsUseCase
from src.application.use_cases.rerank_context import RerankContextUseCase
from src.application.use_cases.retrieve_context import RetrieveContextUseCase
from src.config import get_settings
from src.domain.entities.retrieval import HybridRetrieverConfig, RerankedChunkResult
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker
from src.infrastructure.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from src.infrastructure.llms.qwen_generator import QwenGenerator
from src.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from src.infrastructure.vectorstores.faiss_hnsw_store import FAISSHNSWStore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _check_python_runtime_for_native_stack() -> None:
    """Aviso temprano: FAISS + torch en 3.13/3.14 suelen provocar segfault (no es bug del script)."""
    v = sys.version_info
    if v < (3, 13):
        return
    allow = os.environ.get("RAG_ALLOW_EXPERIMENTAL_PYTHON", "").lower() in ("1", "true", "yes")
    msg = (
        "Estás usando Python %s.%s. La cadena FAISS + PyTorch + sentence-transformers "
        "suele fallar con segmentation fault en versiones muy nuevas. "
        "Recomendado: Python 3.11 o 3.12, venv limpio y `pip install -r requirements.txt`. "
        "Para omitir este aviso: RAG_ALLOW_EXPERIMENTAL_PYTHON=1"
    ) % (v.major, v.minor)
    if v >= (3, 14):
        logger.error(msg)
    else:
        logger.warning(msg)
    if v >= (3, 14) and not allow:
        logger.error("Abortando para evitar segfault. Cambia de versión de Python o define RAG_ALLOW_EXPERIMENTAL_PYTHON=1.")
        raise SystemExit(2)


SUNAT_URLS: list[str] = [
    "https://www.gob.pe/12274-declarar-y-pagar-impuesto-anual-para-rentas-de-trabajo-4ta-y-5ta-categoria",
    "https://www.sunat.gob.pe/legislacion/superin/2025/000386-2025.pdf",
    "https://www.gob.pe/9511-opciones-de-pago-electronico-de-impuestos-a-la-sunat",
    "https://www.gob.pe/1202-superintendencia-nacional-de-aduanas-y-de-administracion-tributaria-calcular-el-impuesto-a-la-renta-de-primera-categoria",
    "https://www.gob.pe/109657-declarar-y-pagar-la-renta-2025-primera-categoria",
    "https://www.gob.pe/8248-calcular-el-impuesto-a-la-renta-de-segunda-categoria-para-venta-de-valores-mobiliarios-y-ganancias-en-fondos-mutuosy/o",
    "https://www.gob.pe/7319-calcular-el-impuesto-a-la-renta-de-quinta-categoria",
    "https://www.gob.pe/7318-superintendencia-nacional-de-aduanas-y-de-administracion-tributaria-calcular-el-impuesto-de-cuarta-categoriay/o",
    "https://emprender.sunat.gob.pe/declaracion-pagos/pagos/fraccionamiento-deudas",
    "https://emprender.sunat.gob.pe/simuladores/fraccionamiento",
    "https://renta.sunat.gob.pe/sites/default/files/inline-files/Cartilla%20Instrucciones%20Personas%20%281%29_0.pdf",
    "https://renta.sunat.gob.pe/sites/default/files/inline-files/cartilla%20Instrucciones%20Empresa_2_6.pdf",
]

GENERATION_QUERIES: list[str] = [
    "¿Cómo se calcula el impuesto a la renta de quinta categoría?",
    "¿Cómo se declara la renta de primera categoría?",
    "¿Qué opciones existen para el fraccionamiento de deudas?",
    "¿Cómo pagar impuestos electrónicamente a Sunat?",
]

# Respuesta vacía o casi vacía
MIN_ANSWER_CHARS = 25

# Frases normalizadas (sin tildes) que sugieren falta de evidencia (heurística)
_INSUFFICIENT_EVIDENCE_MARKERS: tuple[str, ...] = (
    "no hay evidencia suficiente",
    "no existe evidencia suficiente",
    "no hay suficiente informacion",
    "informacion insuficiente",
    "no se encontro informacion",
    "no encontre informacion",
    "el contexto no contiene",
    "los fragmentos no contienen",
    "no aparece en el contexto",
    "no esta en el contexto",
    "no puedo responder con el contexto",
    "no recuperaron fragmentos",
    "sin evidencia en los fragmentos",
)


def _normalize_for_match(text: str) -> str:
    lowered = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )


def _text_preview(text: str, max_len: int = 220) -> str:
    one = text.replace("\n", " ").strip()
    if len(one) <= max_len:
        return one
    return one[: max_len - 3].rstrip() + "..."


def _answer_quality_flags(answer_text: str) -> dict[str, Any]:
    stripped = answer_text.strip()
    norm = _normalize_for_match(stripped)
    too_short = len(stripped) < MIN_ANSWER_CHARS
    empty = len(stripped) == 0
    claims_insufficient = any(m in norm for m in _INSUFFICIENT_EVIDENCE_MARKERS)
    return {
        "answer_char_count": len(stripped),
        "answer_empty": empty,
        "answer_too_short": too_short,
        "claims_insufficient_evidence_heuristic": claims_insufficient,
    }


def _chunks_to_report_rows(chunks: list[RerankedChunkResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in chunks:
        rows.append(
            {
                "rerank_position": c.rerank_position,
                "chunk_id": c.chunk_id,
                "source": c.source,
                "page": c.page,
                "rerank_score": c.rerank_score,
                "text_preview": _text_preview(c.text),
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validar pipeline RAG SUNAT hasta generación con Qwen.",
    )
    p.add_argument("--top-k", type=int, default=5, help="Contexto final (rerank + generación).")
    p.add_argument(
        "--hybrid-pool",
        type=int,
        default=None,
        help="Candidatos híbridos antes de rerank (default: max(settings, top_k)).",
    )
    p.add_argument(
        "--save-report",
        action="store_true",
        help="Escribir data/processed/generation_report.json",
    )
    p.add_argument(
        "--report-path",
        type=str,
        default="data/processed/generation_report.json",
        help="Ruta del JSON de salida.",
    )
    return p.parse_args()


def main() -> int:
    _check_python_runtime_for_native_stack()
    args = _parse_args()
    top_k: int = args.top_k
    if top_k <= 0:
        logger.error("--top-k debe ser > 0.")
        return 1

    settings = get_settings()
    hybrid_pool = args.hybrid_pool if args.hybrid_pool is not None else max(settings.hybrid_final_top_k, top_k)

    logger.info("Cargando documentos (%s URLs)...", len(SUNAT_URLS))
    docs = LoadDocumentsUseCase().execute(SUNAT_URLS)
    chunker = RecursiveChunker(min_chunk_length=settings.default_min_chunk_length)
    chunks = ChunkDocumentsUseCase(chunker).execute(
        docs,
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
        min_chunk_length=settings.default_min_chunk_length,
    )
    logger.info("Chunks: %s", len(chunks))
    if not chunks:
        logger.error("Sin chunks: revisa URLs o cache local.")
        return 1

    embedder = SentenceTransformerEmbeddings(model_name=settings.embedding_model_name)
    t_embed = time.perf_counter()
    embedded = EmbedChunksUseCase(embedder).execute(chunks)
    logger.info("Embeddings: %s en %.2f s.", len(embedded), time.perf_counter() - t_embed)

    dim = len(embedded[0].embedding) if embedded else 384
    store = FAISSHNSWStore(
        dimension=dim,
        m=settings.default_hnsw_m,
        ef_construction=settings.default_hnsw_ef_construction,
        ef_search=settings.default_hnsw_ef_search,
    )
    BuildVectorIndexUseCase(store).execute(embedded)
    bm25 = BuildBm25IndexUseCase().execute(chunks)

    base_h = RetrieveContextUseCase.default_hybrid_config()
    hcfg: HybridRetrieverConfig = replace(
        base_h,
        top_k_dense=max(base_h.top_k_dense, hybrid_pool),
        top_k_sparse=max(base_h.top_k_sparse, hybrid_pool),
        final_top_k=hybrid_pool,
    )
    hybrid_engine = HybridRetriever(hcfg)
    retrieve_uc = RetrieveContextUseCase(
        embedding_adapter=embedder,
        vector_store=store,
        bm25_retriever=bm25,
        hybrid_retriever=hybrid_engine,
    )

    logger.info("Cargando cross-encoder: %s", settings.reranker_model_name)
    rerank_uc = RerankContextUseCase(reranker=CrossEncoderReranker())

    logger.info("Cargando modelo de generación: %s", settings.generation_model_name)
    gen_uc = GenerateAnswerUseCase(llm=QwenGenerator())

    query_reports: list[dict[str, Any]] = []

    for question in GENERATION_QUERIES:
        logger.info("")
        logger.info("################################################################")
        logger.info("PREGUNTA: %s", question)

        t_h = time.perf_counter()
        hybrid_hits = retrieve_uc.execute(question)
        hybrid_seconds = time.perf_counter() - t_h

        t_r = time.perf_counter()
        reranked = rerank_uc.execute(question, hybrid_hits, top_k=top_k)
        rerank_seconds = time.perf_counter() - t_r

        t_g = time.perf_counter()
        answer = gen_uc.execute(question, reranked)
        generation_seconds = time.perf_counter() - t_g

        flags = _answer_quality_flags(answer.text)
        chunk_rows = _chunks_to_report_rows(reranked)

        logger.info("--- Retrieval híbrido: %.4f s | Rerank: %.4f s | Generación: %.4f s ---", hybrid_seconds, rerank_seconds, generation_seconds)
        logger.info("--- Contexto usado (top-%s) ---", len(reranked))
        for row in chunk_rows:
            logger.info(
                "  [#%s] chunk_id=%s score=%.4f",
                row["rerank_position"],
                row["chunk_id"],
                float(row["rerank_score"]),
            )
            logger.info("      source=%s", row["source"])
            logger.info("      page=%s", row["page"])
            logger.info("      preview=%s", row["text_preview"])

        logger.info("--- Respuesta ---\n%s", answer.text)
        logger.info(
            "--- Heurística --- vacía=%s corta=%s posible_falta_evidencia=%s (chars=%s) ---",
            flags["answer_empty"],
            flags["answer_too_short"],
            flags["claims_insufficient_evidence_heuristic"],
            flags["answer_char_count"],
        )

        query_reports.append(
            {
                "question": question,
                "hybrid_retrieval_seconds": hybrid_seconds,
                "reranking_seconds": rerank_seconds,
                "generation_seconds": generation_seconds,
                "context_chunks": chunk_rows,
                "answer_text": answer.text,
                "metadata_answer_keys": list(answer.metadata.keys()),
                "quality_flags": flags,
            }
        )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
        "generation_model": settings.generation_model_name,
        "top_k_context": top_k,
        "hybrid_pool": hybrid_pool,
        "pipeline": {
            "urls": len(SUNAT_URLS),
            "documents": len(docs),
            "chunks": len(chunks),
            "embeddings": len(embedded),
            "faiss_vectors": store.vector_count,
        },
        "queries": query_reports,
        "quality_summary": {
            "any_empty": any(q["quality_flags"]["answer_empty"] for q in query_reports),
            "any_too_short": any(q["quality_flags"]["answer_too_short"] for q in query_reports),
            "any_claims_insufficient": any(
                q["quality_flags"]["claims_insufficient_evidence_heuristic"] for q in query_reports
            ),
        },
    }

    logger.info("")
    logger.info("======== Resumen calidad respuestas ========")
    logger.info("  Alguna vacía: %s", summary["quality_summary"]["any_empty"])
    logger.info("  Alguna demasiado corta: %s", summary["quality_summary"]["any_too_short"])
    logger.info("  Alguna con posible falta de evidencia (heurística): %s", summary["quality_summary"]["any_claims_insufficient"])

    if args.save_report:
        out_path = PROJECT_ROOT / args.report_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Reporte guardado en %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
