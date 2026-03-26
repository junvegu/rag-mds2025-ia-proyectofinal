#!/usr/bin/env python3
"""Validación end-to-end del RAG SUNAT: corpus, índice, evaluación y reporte consolidado.

Ejecuta el pipeline completo (carga → chunk → embed → FAISS + BM25 → retrieve → rerank →
generación con citas) para cada pregunta del evaluation set, calcula métricas automáticas
(ROUGE, BLEU, grounding) y escribe ``data/processed/evaluation_report.json`` listo para
README, notebook o presentación.

Ejemplo:
  python scripts/test_evaluation.py
  python scripts/test_evaluation.py --top-k 5 --no-save-report

Requiere el mismo entorno que la generación (embeddings, FAISS, Qwen).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
from src.application.use_cases.cite_answer import CiteAnswerUseCase
from src.application.use_cases.embed_chunks import EmbedChunksUseCase
from src.application.use_cases.evaluate_rag import EvaluateRagUseCase
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.load_documents import LoadDocumentsUseCase
from src.application.use_cases.rerank_context import RerankContextUseCase
from src.application.use_cases.retrieve_context import RetrieveContextUseCase
from src.config import Settings, get_settings
from src.domain.entities.document import Document
from src.domain.entities.retrieval import HybridRetrieverConfig
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker
from src.infrastructure.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from src.infrastructure.evaluation.bleu_evaluator import BleuEvaluator
from src.infrastructure.evaluation.evaluation_dataset_loader import load_eval_questions
from src.infrastructure.evaluation.rouge_evaluator import RougeEvaluator
from src.infrastructure.llms.qwen_generator import QwenGenerator
from src.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from src.infrastructure.vectorstores.faiss_hnsw_store import FAISSHNSWStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = PROJECT_ROOT / "data/eval/eval_questions.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data/processed/evaluation_report.json"

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validación end-to-end RAG SUNAT: pipeline completo + métricas + JSON consolidado.",
    )
    p.add_argument("--top-k", type=int, default=5, help="Top-k tras rerank por pregunta.")
    p.add_argument("--hybrid-pool", type=int, default=None, help="Pool híbrido antes del rerank (default: settings).")
    p.add_argument("--eval-file", type=str, default=str(DEFAULT_EVAL_PATH))
    p.add_argument(
        "--no-save-report",
        action="store_true",
        help="No escribir evaluation_report.json (solo consola).",
    )
    p.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    return p.parse_args()


def _corpus_stats(seed_urls: list[str], docs: list[Document], chunks_count: int) -> dict[str, Any]:
    unique_sources = len({d.source.strip() for d in docs if d.source and d.source.strip()})
    return {
        "total_seed_urls": len(seed_urls),
        "total_documents": len(docs),
        "unique_document_sources": unique_sources,
        "total_chunks": chunks_count,
    }


def _faiss_descriptor(settings: Settings, dimension: int) -> dict[str, Any]:
    return {
        "implementation": "FAISS IndexHNSWFlat (en memoria, construido en esta corrida)",
        "dimension": dimension,
        "hnsw_m": settings.default_hnsw_m,
        "hnsw_ef_construction": settings.default_hnsw_ef_construction,
        "hnsw_ef_search": settings.default_hnsw_ef_search,
    }


def _retrieval_descriptor(hcfg: HybridRetrieverConfig) -> dict[str, Any]:
    return {
        "mode": "híbrido_denso_sparse",
        "label_es": (
            f"Denso (FAISS HNSW, top_k={hcfg.top_k_dense}) + BM25 Okapi (top_k={hcfg.top_k_sparse}); "
            f"fusión pesos denso={hcfg.dense_weight:.3f} / sparse={hcfg.sparse_weight:.3f}, "
            f"normalización={hcfg.score_normalization}, temperatura={hcfg.fusion_temperature}, "
            f"pool fusionado top_k={hcfg.final_top_k}"
        ),
        "top_k_dense": hcfg.top_k_dense,
        "top_k_sparse": hcfg.top_k_sparse,
        "final_top_k": hcfg.final_top_k,
        "dense_weight": hcfg.dense_weight,
        "sparse_weight": hcfg.sparse_weight,
        "score_normalization": hcfg.score_normalization,
        "fusion_temperature": hcfg.fusion_temperature,
    }


def _build_executive_summary(
    *,
    corpus: dict[str, Any],
    settings: Settings,
    faiss: dict[str, Any],
    retrieval: dict[str, Any],
    reranker_model: str,
    generation_model: str,
    global_metrics: dict[str, Any],
    wall_clock_seconds: float,
    top_k: int,
    hybrid_pool: int,
) -> dict[str, Any]:
    """Bloque único para JSON, README y presentación."""
    return {
        "run_purpose": "validacion_end_to_end_rag_sunat",
        "corpus": corpus,
        "models_and_index": {
            "embedding_model": settings.embedding_model_name,
            "faiss_index": faiss,
            "retrieval": retrieval,
            "reranker_model": reranker_model,
            "generation_model": generation_model,
        },
        "evaluation_aggregates": {
            "avg_rouge_1": global_metrics["avg_rouge_1"],
            "avg_rouge_l": global_metrics["avg_rouge_l"],
            "avg_grounding_score": global_metrics["avg_grounding_score"],
            "total_hallucination_flags": global_metrics["total_hallucination_flags"],
            "total_questions": global_metrics["total_questions"],
            "avg_bleu": global_metrics.get("avg_bleu"),
            "hallucination_rate": global_metrics.get("hallucination_rate"),
        },
        "run_parameters": {
            "rerank_top_k": top_k,
            "hybrid_pool": hybrid_pool,
            "wall_clock_seconds_evaluation_phase": round(wall_clock_seconds, 4),
        },
    }


def _print_evaluation_block(report: Any, wall_clock: float) -> None:
    ser = report.to_serializable()
    gm = ser["global_metrics"]
    con = ser["conclusion"]
    print("\n" + "=" * 72)
    print("EVALUACIÓN AUTOMÁTICA (por pregunta: retrieve → rerank → generación)")
    print("=" * 72)
    print(f"  total_questions              : {gm['total_questions']}")
    print(f"  tiempo fase evaluación (wall): {wall_clock:.1f} s")
    print(f"  avg_rouge_1                  : {gm['avg_rouge_1']:.4f}")
    print(f"  avg_rouge_l                  : {gm['avg_rouge_l']:.4f}")
    print(f"  avg_bleu (0–100)             : {gm['avg_bleu']:.2f}")
    print(f"  avg_grounding_score          : {gm['avg_grounding_score']}")
    print(f"  total_hallucination_flags    : {gm['total_hallucination_flags']}")
    print(f"  hallucination_rate           : {gm['hallucination_rate']:.2%}")
    print("-" * 72)
    print(f"  Conclusión automática: {con['verdict_es']}")
    print(f"  {con['summary_es']}")
    print("-" * 72)
    print("  Nota:", report.disclaimer[:120] + "…")
    print("=" * 72 + "\n")


def _print_executive_summary(summary: dict[str, Any], report_path: Path, *, will_save: bool) -> None:
    c = summary["corpus"]
    m = summary["models_and_index"]
    e = summary["evaluation_aggregates"]
    print("\n" + "=" * 72)
    print("RESUMEN EJECUTIVO — Sistema RAG SUNAT (cierre técnico)")
    print("=" * 72)
    print("  Corpus")
    print(f"    Total URLs semilla (ingesta)     : {c['total_seed_urls']}")
    print(f"    Total documentos cargados        : {c['total_documents']}")
    print(f"    Fuentes únicas (documentos)      : {c['unique_document_sources']}")
    print(f"    Total chunks indexados           : {c['total_chunks']}")
    print("  Modelos e índice")
    print(f"    Modelo de embeddings             : {m['embedding_model']}")
    f = m["faiss_index"]
    print(
        f"    Índice FAISS                     : {f['implementation']} | dim={f['dimension']}, "
        f"M={f['hnsw_m']}, efConstruction={f['hnsw_ef_construction']}, efSearch={f['hnsw_ef_search']}"
    )
    print(f"    Recuperación                     : {m['retrieval']['label_es']}")
    print(f"    Reranker (cross-encoder)         : {m['reranker_model']}")
    print(f"    Modelo generativo                : {m['generation_model']}")
    print("  Métricas globales (evaluation set)")
    print(f"    avg_rouge_1                      : {e['avg_rouge_1']:.4f}")
    print(f"    avg_rouge_l                      : {e['avg_rouge_l']:.4f}")
    g = e["avg_grounding_score"]
    print(f"    avg_grounding_score              : {g if g is not None else 'N/D'}")
    print(f"    total_hallucination_flags        : {e['total_hallucination_flags']}")
    print("=" * 72)
    if will_save:
        print(
            f"  JSON consolidado escrito en {report_path} (clave «executive_summary» + métricas completas)."
        )
    else:
        print(f"  Sin guardar JSON (--no-save-report). Ruta prevista: {report_path}")
    print("=" * 72 + "\n")


def main() -> int:
    if sys.version_info >= (3, 14):
        if os.environ.get("RAG_ALLOW_EXPERIMENTAL_PYTHON", "").lower() not in ("1", "true", "yes"):
            logger.error("Python 3.14+ no recomendado; define RAG_ALLOW_EXPERIMENTAL_PYTHON=1 o usa 3.12.")
            return 2

    args = _parse_args()
    if args.top_k <= 0:
        return 1

    settings = get_settings()
    hybrid_pool = args.hybrid_pool if args.hybrid_pool is not None else max(settings.hybrid_final_top_k, args.top_k)

    eval_path = Path(args.eval_file)
    if not eval_path.is_file():
        logger.error("No existe archivo de evaluación: %s", eval_path)
        return 1

    items = load_eval_questions(eval_path)
    if not items:
        logger.error("Sin ítems en el JSON de evaluación.")
        return 1

    logger.info("=== Fase 1: ingesta y construcción de índices ===")
    logger.info("Cargando corpus (%s URLs semilla)...", len(SUNAT_URLS))
    docs = LoadDocumentsUseCase().execute(SUNAT_URLS)
    chunker = RecursiveChunker(min_chunk_length=settings.default_min_chunk_length)
    chunks = ChunkDocumentsUseCase(chunker).execute(
        docs,
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
        min_chunk_length=settings.default_min_chunk_length,
    )
    if not chunks:
        logger.error("Sin chunks.")
        return 1

    corpus = _corpus_stats(SUNAT_URLS, docs, len(chunks))

    embedder = SentenceTransformerEmbeddings(model_name=settings.embedding_model_name)
    embedded = EmbedChunksUseCase(embedder).execute(chunks)
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
    rerank_uc = RerankContextUseCase(reranker=CrossEncoderReranker())
    gen_uc = GenerateAnswerUseCase(
        llm=QwenGenerator(),
        citation_use_case=CiteAnswerUseCase(embedder),
    )

    faiss_desc = _faiss_descriptor(settings, dim)
    retrieval_desc = _retrieval_descriptor(hcfg)

    logger.info("=== Fase 2: evaluación end-to-end (%s preguntas) ===", len(items))
    eval_uc = EvaluateRagUseCase(
        retrieve=retrieve_uc,
        rerank=rerank_uc,
        generate=gen_uc,
        rouge=RougeEvaluator(),
        bleu=BleuEvaluator(),
        top_k=args.top_k,
    )

    t0 = time.perf_counter()
    report = eval_uc.execute(items)
    wall = time.perf_counter() - t0

    ser = report.to_serializable()
    gm = ser["global_metrics"]

    executive_summary = _build_executive_summary(
        corpus=corpus,
        settings=settings,
        faiss=faiss_desc,
        retrieval=retrieval_desc,
        reranker_model=settings.reranker_model_name,
        generation_model=settings.generation_model_name,
        global_metrics=gm,
        wall_clock_seconds=wall,
        top_k=args.top_k,
        hybrid_pool=hybrid_pool,
    )

    _print_evaluation_block(report, wall)
    out = Path(args.report_path)
    will_save = not args.no_save_report
    _print_executive_summary(executive_summary, out, will_save=will_save)

    if will_save:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "report_path": str(out.resolve()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_type": "end_to_end_validation",
            "eval_file": str(eval_path),
            "seed_urls": list(SUNAT_URLS),
            "top_k": args.top_k,
            "hybrid_pool": hybrid_pool,
            "wall_clock_seconds_evaluation": wall,
            "embedding_model": settings.embedding_model_name,
            "generation_model": settings.generation_model_name,
            "reranker_model": settings.reranker_model_name,
            "chunking": {
                "chunk_size": settings.default_chunk_size,
                "chunk_overlap": settings.default_chunk_overlap,
                "min_chunk_length": settings.default_min_chunk_length,
            },
            "executive_summary": executive_summary,
            **ser,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Reporte consolidado guardado en %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
