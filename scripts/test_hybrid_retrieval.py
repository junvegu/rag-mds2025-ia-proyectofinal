"""Validación SUNAT: comparar retrieval denso (FAISS), BM25 e híbrido; reporte JSON opcional."""

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
from src.application.use_cases.retrieve_context import RetrieveContextUseCase
from src.config import get_settings
from src.domain.entities.retrieval import HybridChunkResult, HybridRetrieverConfig, RetrievedChunk
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker
from src.infrastructure.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from src.infrastructure.retrieval.bm25_retriever import Bm25Hit
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

# Consultas obligatorias del usuario + frases objetivo para medir alineación léxica
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

ConclusionKind = Literal["dense_mejor", "hybrid_mejor", "empate", "requiere_reranker"]

_CONCLUSION_ES: dict[ConclusionKind, str] = {
    "dense_mejor": "denso mejor",
    "hybrid_mejor": "híbrido mejor",
    "empate": "empate",
    "requiere_reranker": "requiere reranker",
}


@dataclass(slots=True)
class KeywordAudit:
    """Conteo de aciertos léxicos esperados en el top-k (texto normalizado)."""

    keyword_hits_in_topk: int
    keywords_matched: list[str]
    first_relevant_rank: int | None  # 1-based; None si ningún hit contiene frase objetivo


@dataclass(slots=True)
class QueryValidationBlock:
    query: str
    expected_keyword_phrases: list[str]
    dense_seconds: float
    sparse_seconds: float
    hybrid_seconds: float
    dense_results: list[dict[str, Any]]
    sparse_results: list[dict[str, Any]]
    hybrid_results: list[dict[str, Any]]
    dense_audit: dict[str, Any]
    sparse_audit: dict[str, Any]
    hybrid_audit: dict[str, Any]
    conclusion: ConclusionKind
    conclusion_note: str


def _strip_accents(text: str) -> str:
    """Minúsculas y sin tildes para comparar con corpus heterogéneo."""
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
    """Cuántos del top-k contienen al menos una frase clave; primera posición útil."""
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


def _conclusion_from_audits(
    dense_audit: KeywordAudit,
    hybrid_audit: KeywordAudit,
    top_k: int,
) -> tuple[ConclusionKind, str]:
    """Regla simple defendible para slides (sin métricas externas)."""
    d_h, h_h = dense_audit.keyword_hits_in_topk, hybrid_audit.keyword_hits_in_topk
    d_fr, h_fr = dense_audit.first_relevant_rank, hybrid_audit.first_relevant_rank

    weak = d_h == 0 and h_h == 0
    if weak:
        return "requiere_reranker", "Ningún resultado del top-k contiene frases clave esperadas en denso ni híbrido."

    if h_h > d_h:
        return "hybrid_mejor", f"Más chunks con términos clave en híbrido ({h_h} vs {d_h} en top-{top_k})."
    if d_h > h_h:
        return "dense_mejor", f"Más chunks con términos clave en denso ({d_h} vs {h_h} en top-{top_k})."

    # empate en conteo: desempate por primera posición relevante
    if h_fr is not None and d_fr is not None:
        if h_fr < d_fr:
            return "hybrid_mejor", f"Mismo conteo ({d_h}); híbrido ubica antes el primer hit relevante (ranks {h_fr} vs {d_fr})."
        if d_fr < h_fr:
            return "dense_mejor", f"Mismo conteo ({d_h}); denso ubica antes el primer hit relevante (ranks {d_fr} vs {h_fr})."

    if d_h >= max(1, (top_k + 1) // 2):
        return "empate", f"Ambos cubren bien términos clave ({d_h} hits en top-{top_k})."

    return "requiere_reranker", f"Pocos hits léxicos en ambos ({d_h}); conviene reranker o ampliar corpus."


def _dense_to_rows(dense: list[RetrievedChunk]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in dense:
        rows.append(
            {
                "chunk_id": item.chunk.chunk_id,
                "dense_score": item.dense_score,
                "sparse_score": None,
                "hybrid_score": None,
                "source": item.chunk.source,
                "page": item.chunk.page,
                "text_preview": _text_preview(item.chunk.text),
            }
        )
    return rows


def _sparse_to_rows(sparse: list[Bm25Hit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in sparse:
        rows.append(
            {
                "chunk_id": hit.chunk_id,
                "dense_score": None,
                "sparse_score": hit.score,
                "hybrid_score": None,
                "source": hit.source,
                "page": hit.page,
                "text_preview": _text_preview(hit.text),
            }
        )
    return rows


def _hybrid_to_rows(hybrid: list[HybridChunkResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h in hybrid:
        rows.append(
            {
                "chunk_id": h.chunk_id,
                "dense_score": h.dense_score,
                "sparse_score": h.sparse_score,
                "hybrid_score": h.hybrid_score,
                "source": h.source,
                "page": h.page,
                "text_preview": _text_preview(h.text),
            }
        )
    return rows


def _audit_to_dict(a: KeywordAudit, phrases: list[str]) -> dict[str, Any]:
    return {
        "expected_phrases": phrases,
        "keyword_hits_in_topk": a.keyword_hits_in_topk,
        "keywords_matched": a.keywords_matched,
        "first_relevant_rank": a.first_relevant_rank,
    }


def _print_block(title: str, rows: list[dict[str, Any]], top_k: int) -> None:
    logger.info("======== %s ========", title)
    for i, row in enumerate(rows[:top_k], start=1):
        hs = row.get("hybrid_score")
        ds = row.get("dense_score")
        ss = row.get("sparse_score")
        logger.info(
            "  [%s] hybrid=%s dense=%s sparse=%s",
            i,
            f"{hs:.6f}" if hs is not None else "—",
            f"{ds:.6f}" if ds is not None else "—",
            f"{ss:.6f}" if ss is not None else "—",
        )
        logger.info("      source=%s", row.get("source"))
        logger.info("      page=%s", row.get("page"))
        logger.info("      preview=%s", row.get("text_preview"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validar retrieval híbrido SUNAT vs denso y BM25.")
    p.add_argument("--top-k", type=int, default=5, help="Top-k a imprimir y auditar por canal.")
    p.add_argument(
        "--save-report",
        action="store_true",
        help="Escribir data/processed/hybrid_retrieval_report.json",
    )
    p.add_argument(
        "--report-path",
        type=str,
        default="data/processed/hybrid_retrieval_report.json",
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

    embedder = SentenceTransformerEmbeddings(model_name=settings.embedding_model_name)
    t0 = time.perf_counter()
    embedded = EmbedChunksUseCase(embedder).execute(chunks)
    logger.info("Embeddings listos en %.2f s (%s vectores).", time.perf_counter() - t0, len(embedded))

    dim = len(embedded[0].embedding) if embedded else 384
    store = FAISSHNSWStore(
        dimension=dim,
        m=settings.default_hnsw_m,
        ef_construction=settings.default_hnsw_ef_construction,
        ef_search=settings.default_hnsw_ef_search,
    )
    BuildVectorIndexUseCase(store).execute(embedded)
    bm25 = BuildBm25IndexUseCase().execute(chunks)

    base_cfg = RetrieveContextUseCase.default_hybrid_config()
    hcfg: HybridRetrieverConfig = replace(
        base_cfg,
        top_k_dense=max(base_cfg.top_k_dense, top_k),
        top_k_sparse=max(base_cfg.top_k_sparse, top_k),
        final_top_k=top_k,
    )
    hybrid_engine = HybridRetriever(hcfg)
    retrieve = RetrieveContextUseCase(
        embedding_adapter=embedder,
        vector_store=store,
        bm25_retriever=bm25,
        hybrid_retriever=hybrid_engine,
    )

    report_queries: list[dict[str, Any]] = []

    for query_text, expected_phrases in VALIDATION_QUERIES:
        logger.info("")
        logger.info("################################################################")
        logger.info("CONSULTA: %s", query_text)
        logger.info("Frases clave esperadas (cualquiera cuenta): %s", expected_phrases)

        stripped = query_text.strip()
        qv = embedder.embed_query(stripped)

        t_dense = time.perf_counter()
        dense_hits = store.search(qv, top_k)
        dense_seconds = time.perf_counter() - t_dense

        t_sparse = time.perf_counter()
        sparse_hits = bm25.search(stripped, top_k)
        sparse_seconds = time.perf_counter() - t_sparse

        t_hyb = time.perf_counter()
        hybrid_hits = retrieve.execute(query_text)
        hybrid_seconds = time.perf_counter() - t_hyb

        dense_rows = _dense_to_rows(dense_hits)
        sparse_rows = _sparse_to_rows(sparse_hits)
        hybrid_rows = _hybrid_to_rows(hybrid_hits)

        dense_texts = [item.chunk.text for item in dense_hits]
        hybrid_texts = [h.text for h in hybrid_hits]
        sparse_texts = [h.text for h in sparse_hits]

        dense_audit = _audit_keywords(dense_texts, expected_phrases)
        hybrid_audit = _audit_keywords(hybrid_texts, expected_phrases)
        sparse_audit = _audit_keywords(sparse_texts, expected_phrases)

        conclusion, note = _conclusion_from_audits(dense_audit, hybrid_audit, top_k)

        _print_block(f"FAISS denso (top-{top_k})", dense_rows, top_k)
        _print_block(f"BM25 (top-{top_k})", sparse_rows, top_k)
        _print_block(f"Híbrido (top-{top_k})", hybrid_rows, top_k)

        logger.info("--- Auditoría léxica (top-%s) ---", top_k)
        logger.info(
            "  Denso:  hits=%s matched=%s first_rank=%s",
            dense_audit.keyword_hits_in_topk,
            dense_audit.keywords_matched,
            dense_audit.first_relevant_rank,
        )
        logger.info(
            "  BM25:   hits=%s matched=%s first_rank=%s",
            sparse_audit.keyword_hits_in_topk,
            sparse_audit.keywords_matched,
            sparse_audit.first_relevant_rank,
        )
        logger.info(
            "  Híbrido: hits=%s matched=%s first_rank=%s",
            hybrid_audit.keyword_hits_in_topk,
            hybrid_audit.keywords_matched,
            hybrid_audit.first_relevant_rank,
        )
        logger.info(
            ">>> CONCLUSIÓN: %s — %s",
            _CONCLUSION_ES.get(conclusion, conclusion),
            note,
        )

        block = QueryValidationBlock(
            query=query_text,
            expected_keyword_phrases=expected_phrases,
            dense_seconds=dense_seconds,
            sparse_seconds=sparse_seconds,
            hybrid_seconds=hybrid_seconds,
            dense_results=dense_rows,
            sparse_results=sparse_rows,
            hybrid_results=hybrid_rows,
            dense_audit=_audit_to_dict(dense_audit, expected_phrases),
            sparse_audit=_audit_to_dict(sparse_audit, expected_phrases),
            hybrid_audit=_audit_to_dict(hybrid_audit, expected_phrases),
            conclusion=conclusion,
            conclusion_note=note,
        )
        report_queries.append(asdict(block))

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.embedding_model_name,
        "top_k": top_k,
        "hybrid_config": {
            "dense_weight": hcfg.dense_weight,
            "sparse_weight": hcfg.sparse_weight,
            "score_normalization": hcfg.score_normalization,
            "fusion_temperature": hcfg.fusion_temperature,
            "top_k_dense": hcfg.top_k_dense,
            "top_k_sparse": hcfg.top_k_sparse,
            "final_top_k": hcfg.final_top_k,
        },
        "pipeline": {
            "urls": len(SUNAT_URLS),
            "documents": len(docs),
            "chunks": len(chunks),
            "embeddings": len(embedded),
            "faiss_vectors": store.vector_count,
        },
        "queries": report_queries,
        "conclusion_counts": _count_conclusions(report_queries),
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
    counts: dict[str, int] = {
        "dense_mejor": 0,
        "hybrid_mejor": 0,
        "empate": 0,
        "requiere_reranker": 0,
    }
    for q in queries:
        c = q.get("conclusion", "")
        if c in counts:
            counts[c] += 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
