"""Validación SUNAT: orden híbrido vs tras cross-encoder; reporte JSON opcional."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime_env import apply_darwin_openmp_mitigations

apply_darwin_openmp_mitigations()

from src.application.use_cases.build_bm25_index import BuildBm25IndexUseCase
from src.application.use_cases.build_vector_index import BuildVectorIndexUseCase
from src.application.use_cases.chunk_documents import ChunkDocumentsUseCase
from src.application.use_cases.embed_chunks import EmbedChunksUseCase
from src.application.use_cases.load_documents import LoadDocumentsUseCase
from src.application.use_cases.rerank_context import RerankContextUseCase
from src.application.use_cases.retrieve_context import RetrieveContextUseCase
from src.config import get_settings
from src.domain.entities.retrieval import HybridChunkResult, HybridRetrieverConfig, RerankedChunkResult
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker
from src.infrastructure.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from src.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from src.infrastructure.vectorstores.faiss_hnsw_store import FAISSHNSWStore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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

VALIDATION_QUERIES: list[tuple[str, list[str]]] = [
    (
        "¿Cómo se calcula el impuesto a la renta de quinta categoría?",
        ["quinta categoría", "quinta categoria", "5ta", "5ta categoria"],
    ),
    (
        "¿Cómo se declara la renta de primera categoría?",
        ["primera categoría", "primera categoria", "1ra categoria"],
    ),
    (
        "¿Qué opciones existen para el fraccionamiento de deudas?",
        ["fraccionamiento", "aplazamiento", "deuda tributaria"],
    ),
    (
        "¿Cómo pagar impuestos electrónicamente a Sunat?",
        [
            "pago electrónico",
            "pago electronico",
            "electronico",
            "banca por internet",
            "transferencia de fondos",
        ],
    ),
]

RerankConclusionKind = Literal[
    "reranker_mejora",
    "empate",
    "no_mejora",
    "requiere_revision_manual",
]

_CONCLUSION_ES: dict[RerankConclusionKind, str] = {
    "reranker_mejora": "reranker mejora",
    "empate": "empate",
    "no_mejora": "no mejora",
    "requiere_revision_manual": "requiere revisión manual",
}


@dataclass(slots=True)
class KeywordAudit:
    keyword_hits_in_topk: int
    keywords_matched: list[str]
    first_relevant_rank: int | None


def _strip_accents(text: str) -> str:
    lowered = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )


def _text_preview(text: str, max_len: int = 200) -> str:
    one = text.replace("\n", " ").strip()
    if len(one) <= max_len:
        return one
    return one[: max_len - 3].rstrip() + "..."


def _audit_keywords(texts: list[str], phrases: list[str]) -> KeywordAudit:
    norm_phrases = [_strip_accents(p) for p in phrases]
    hits = 0
    matched: set[str] = set()
    first_rank: int | None = None
    for rank, raw in enumerate(texts, start=1):
        blob = _strip_accents(raw)
        ok = False
        for phrase, orig in zip(norm_phrases, phrases):
            if phrase in blob:
                ok = True
                matched.add(orig)
        if ok:
            hits += 1
            if first_rank is None:
                first_rank = rank
    return KeywordAudit(
        keyword_hits_in_topk=hits,
        keywords_matched=sorted(matched),
        first_relevant_rank=first_rank,
    )


def _audit_to_dict(a: KeywordAudit, phrases: list[str]) -> dict[str, Any]:
    return {
        "expected_phrases": phrases,
        "keyword_hits_in_topk": a.keyword_hits_in_topk,
        "keywords_matched": a.keywords_matched,
        "first_relevant_rank": a.first_relevant_rank,
    }


def _audit_sort_key(audit: KeywordAudit, top_k: int) -> tuple[int, int]:
    """Mayor es mejor: (hits en top-k, −rank del primer hit; sin hit → rank peor ficticio)."""
    fr = audit.first_relevant_rank if audit.first_relevant_rank is not None else top_k + 1
    return (audit.keyword_hits_in_topk, -fr)


def _conclusion_rerank_vs_hybrid(
    before: KeywordAudit,
    after: KeywordAudit,
    top_k: int,
) -> tuple[RerankConclusionKind, str]:
    """Orden lexicográfico defendible: primero #hits con frase clave, luego rank del primer hit."""
    if before.keyword_hits_in_topk == 0 and after.keyword_hits_in_topk == 0:
        return (
            "requiere_revision_manual",
            f"Sin coincidencias léxicas en top-{top_k} antes ni después; revisar frases o corpus.",
        )

    kb = _audit_sort_key(before, top_k)
    ka = _audit_sort_key(after, top_k)
    if ka > kb:
        return (
            "reranker_mejora",
            f"Mejor según heurística (hits, posición primer hit): {ka} vs {kb}.",
        )
    if ka < kb:
        return (
            "no_mejora",
            f"Peor según heurística (hits, posición primer hit): {ka} vs {kb}.",
        )
    return (
        "empate",
        f"Misma heurística antes y después: {ka} (hits y rank del primer hit equivalentes).",
    )


def _hybrid_rows(hybrid: list[HybridChunkResult], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, h in enumerate(hybrid[:top_k], start=1):
        rows.append(
            {
                "rank": i,
                "chunk_id": h.chunk_id,
                "rerank_score": None,
                "hybrid_score": h.hybrid_score,
                "dense_score": h.dense_score,
                "sparse_score": h.sparse_score,
                "source": h.source,
                "page": h.page,
                "text_preview": _text_preview(h.text),
            }
        )
    return rows


def _reranked_rows(reranked: list[RerankedChunkResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in reranked:
        rows.append(
            {
                "rank": r.rerank_position,
                "chunk_id": r.chunk_id,
                "rerank_score": r.rerank_score,
                "hybrid_score": r.hybrid_score,
                "dense_score": r.dense_score,
                "sparse_score": r.sparse_score,
                "source": r.source,
                "page": r.page,
                "text_preview": _text_preview(r.text),
            }
        )
    return rows


def _print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    logger.info("======== %s ========", title)
    for row in rows:
        rs = row.get("rerank_score")
        logger.info(
            "  [rank=%s] rerank=%s hybrid=%.6f dense=%s sparse=%s",
            row["rank"],
            f"{rs:.6f}" if rs is not None else "—",
            float(row["hybrid_score"]),
            f"{row['dense_score']:.6f}" if row.get("dense_score") is not None else "—",
            f"{row['sparse_score']:.6f}" if row.get("sparse_score") is not None else "—",
        )
        logger.info("      source=%s", row.get("source"))
        logger.info("      page=%s", row.get("page"))
        logger.info("      preview=%s", row.get("text_preview"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validar reranking cross-encoder tras retrieval híbrido SUNAT.")
    p.add_argument("--top-k", type=int, default=5, help="Top-k a mostrar y auditar (híbrido truncado y rerank).")
    p.add_argument(
        "--hybrid-pool",
        type=int,
        default=None,
        help="Candidatos que entrega el híbrido antes de rerank (default: max(settings, top_k)).",
    )
    p.add_argument(
        "--save-report",
        action="store_true",
        help="Escribir data/processed/reranker_report.json",
    )
    p.add_argument(
        "--report-path",
        type=str,
        default="data/processed/reranker_report.json",
        help="Ruta del JSON de salida.",
    )
    return p.parse_args()


def main() -> int:
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

    query_reports: list[dict[str, Any]] = []

    for query_text, phrases in VALIDATION_QUERIES:
        logger.info("")
        logger.info("################################################################")
        logger.info("CONSULTA: %s", query_text)

        t_h = time.perf_counter()
        hybrid_full = retrieve_uc.execute(query_text)
        hybrid_seconds = time.perf_counter() - t_h

        t_r = time.perf_counter()
        reranked = rerank_uc.execute(query_text, hybrid_full, top_k=top_k)
        rerank_seconds = time.perf_counter() - t_r

        hybrid_slice = hybrid_full[:top_k]
        texts_before = [h.text for h in hybrid_slice]
        texts_after = [r.text for r in reranked]

        audit_before = _audit_keywords(texts_before, phrases)
        audit_after = _audit_keywords(texts_after, phrases)
        conclusion, note = _conclusion_rerank_vs_hybrid(audit_before, audit_after, top_k)

        h_rows = _hybrid_rows(hybrid_full, top_k)
        r_rows = _reranked_rows(reranked)

        _print_rows(f"Híbrido (antes rerank, top-{top_k})", h_rows)
        _print_rows(f"Tras rerank (top-{top_k})", r_rows)

        logger.info("--- Tiempos --- hybrid_retrieval=%.4f s | reranking=%.4f s", hybrid_seconds, rerank_seconds)
        logger.info(
            "--- Auditoría léxica --- antes: hits=%s first_rank=%s | después: hits=%s first_rank=%s",
            audit_before.keyword_hits_in_topk,
            audit_before.first_relevant_rank,
            audit_after.keyword_hits_in_topk,
            audit_after.first_relevant_rank,
        )
        logger.info(
            ">>> CONCLUSIÓN: %s — %s",
            _CONCLUSION_ES.get(conclusion, conclusion),
            note,
        )

        query_reports.append(
            {
                "query": query_text,
                "expected_keyword_phrases": phrases,
                "hybrid_retrieval_seconds": hybrid_seconds,
                "reranking_seconds": rerank_seconds,
                "hybrid_pool_size": len(hybrid_full),
                "hybrid_before": h_rows,
                "after_rerank": r_rows,
                "audit_before": _audit_to_dict(audit_before, phrases),
                "audit_after": _audit_to_dict(audit_after, phrases),
                "conclusion": conclusion,
                "conclusion_note": note,
            }
        )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
        "top_k_print": top_k,
        "hybrid_pool": hybrid_pool,
        "hybrid_config": asdict(hcfg),
        "pipeline": {
            "urls": len(SUNAT_URLS),
            "documents": len(docs),
            "chunks": len(chunks),
            "embeddings": len(embedded),
            "faiss_vectors": store.vector_count,
        },
        "queries": query_reports,
        "conclusion_counts": _count_conclusions(query_reports),
    }

    logger.info("")
    logger.info("======== Resumen global ========")
    for k, v in summary["conclusion_counts"].items():
        logger.info("  %s: %s", k, v)

    if args.save_report:
        out_path = PROJECT_ROOT / args.report_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Reporte guardado en %s", out_path)

    return 0


def _count_conclusions(queries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "reranker_mejora": 0,
        "empate": 0,
        "no_mejora": 0,
        "requiere_revision_manual": 0,
    }
    for q in queries:
        c = q.get("conclusion", "")
        if c in counts:
            counts[c] += 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
