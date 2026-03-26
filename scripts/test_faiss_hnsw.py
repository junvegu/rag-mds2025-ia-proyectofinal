"""Validate FAISS HNSW index build, persistence, and top-k search."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime_env import apply_darwin_openmp_mitigations

apply_darwin_openmp_mitigations()

from src.application.use_cases.build_vector_index import BuildVectorIndexUseCase
from src.application.use_cases.chunk_documents import ChunkDocumentsUseCase
from src.application.use_cases.embed_chunks import EmbedChunksUseCase
from src.application.use_cases.load_documents import LoadDocumentsUseCase
from src.application.use_cases.search_vector_index import SearchVectorIndexUseCase
from src.config import get_settings
from src.domain.entities.retrieval import RetrievedChunk
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker
from src.infrastructure.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from src.infrastructure.vectorstores.faiss_hnsw_store import FAISSHNSWStore

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

TEST_QUERIES_ES: list[str] = [
    "¿Qué es la renta de primera categoría y cómo se declara?",
    "¿Cómo se calcula el impuesto a la renta de quinta categoría?",
    "¿Qué es el fraccionamiento de deudas tributarias ante SUNAT?",
]


@dataclass(slots=True)
class QuerySearchReport:
    query: str
    top_k: int
    search_seconds: float
    num_results: int
    results: list[dict[str, Any]]


@dataclass(slots=True)
class FaissValidationReport:
    total_documents: int
    total_chunks: int
    total_embeddings: int
    index_vector_count: int
    index_matches_embeddings: bool
    load_save_ok: bool
    all_searches_non_empty: bool
    metadata_consistent: bool
    hnsw_config: dict[str, int]
    embedding_model: str
    queries: list[QuerySearchReport]


def _result_to_dict(item: RetrievedChunk) -> dict[str, Any]:
    preview = item.chunk.text[:200].replace("\n", " ").strip()
    if len(item.chunk.text) > 200:
        preview += "..."
    return {
        "chunk_id": item.chunk.chunk_id,
        "doc_id": item.chunk.doc_id,
        "source": item.chunk.source,
        "page": item.chunk.page,
        "score": item.dense_score,
        "text_preview": preview,
    }


def _metadata_consistent(results: list[RetrievedChunk]) -> bool:
    for item in results:
        if not item.chunk.chunk_id or not item.chunk.doc_id or not item.chunk.source:
            return False
        if not item.chunk.text or not item.chunk.text.strip():
            return False
        if item.dense_score is None:
            return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FAISS HNSW index for SUNAT RAG.")
    parser.add_argument("--top-k", type=int, default=5, help="Neighbors per query.")
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Write data/processed/faiss_report.json",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/processed/faiss_validation_index",
        help="Directory for save/load round-trip test.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()

    logger.info("Loading documents...")
    documents = LoadDocumentsUseCase().execute(SUNAT_URLS)
    chunker = RecursiveChunker(min_chunk_length=settings.default_min_chunk_length)
    chunks = ChunkDocumentsUseCase(chunker=chunker).execute(
        documents=documents,
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
        min_chunk_length=settings.default_min_chunk_length,
    )

    embed_adapter = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model_name,
        batch_size=32,
        normalize_embeddings=True,
    )
    embedded = EmbedChunksUseCase(embedding_adapter=embed_adapter).execute(chunks)

    if not embedded:
        logger.error("No embedded chunks; aborting.")
        return 1

    dim = len(embedded[0].embedding)
    store = FAISSHNSWStore(
        dimension=dim,
        m=settings.default_hnsw_m,
        ef_construction=settings.default_hnsw_ef_construction,
        ef_search=settings.default_hnsw_ef_search,
    )
    build_uc = BuildVectorIndexUseCase(vector_store=store)
    build_uc.execute(embedded_chunks=embedded, save_dir=None)

    index_count = store.vector_count
    emb_count = len(embedded)
    index_matches = index_count == emb_count

    print("=== FAISS HNSW Validation ===")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {emb_count}")
    print(f"Index vectors (ntotal): {index_count}")
    print(f"Index matches embeddings: {index_matches}")
    print(
        f"HNSW: M={settings.default_hnsw_m}, "
        f"ef_construction={settings.default_hnsw_ef_construction}, "
        f"ef_search={settings.default_hnsw_ef_search}"
    )
    print(f"Embedding model: {settings.embedding_model_name}")
    print(f"Vector dimension: {dim}")

    load_save_ok = False
    index_dir = Path(args.index_dir)
    try:
        store.save(str(index_dir))
        loaded = FAISSHNSWStore.load(str(index_dir))
        load_save_ok = loaded.vector_count == emb_count and loaded.metadata_count == emb_count
        if load_save_ok:
            store = loaded
            logger.info("Save/load round-trip OK; using loaded index for queries.")
        else:
            logger.warning("Save/load mismatch; continuing with in-memory index.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Save/load test failed: %s", exc)

    search_uc = SearchVectorIndexUseCase(vector_store=store)
    query_reports: list[QuerySearchReport] = []
    all_non_empty = True
    meta_ok_all = True

    for query in TEST_QUERIES_ES:
        t0 = time.perf_counter()
        try:
            qvec = embed_adapter.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("Query embedding failed: %s", exc)
            all_non_empty = False
            query_reports.append(
                QuerySearchReport(
                    query=query,
                    top_k=args.top_k,
                    search_seconds=0.0,
                    num_results=0,
                    results=[],
                )
            )
            continue

        t1 = time.perf_counter()
        results = search_uc.execute(query_embedding=qvec, top_k=args.top_k)
        t2 = time.perf_counter()
        search_elapsed = t2 - t1
        embed_elapsed = t1 - t0

        if not results:
            all_non_empty = False
        if results and not _metadata_consistent(results):
            meta_ok_all = False

        print(f"\n--- Query ---\n{query}")
        print(f"top_k={args.top_k} | embed_s={embed_elapsed:.4f} | search_s={search_elapsed:.4f}")
        for i, item in enumerate(results, start=1):
            d = _result_to_dict(item)
            src = str(d["source"])
            src_disp = f"{src[:70]}..." if len(src) > 70 else src
            print(f"  {i}. score={d['score']:.4f} | source={src_disp} | page={d['page']}")
            print(f"      chunk_id={d['chunk_id']}")
            print(f"      {d['text_preview']}")

        query_reports.append(
            QuerySearchReport(
                query=query,
                top_k=args.top_k,
                search_seconds=search_elapsed,
                num_results=len(results),
                results=[_result_to_dict(r) for r in results],
            )
        )

    print("\n=== Validaciones ===")
    print(f"ntotal == len(embeddings): {index_matches}")
    print(f"Búsquedas con resultados: {all_non_empty}")
    print(f"Metadata consistente: {meta_ok_all}")
    print(f"save/load OK: {load_save_ok}")

    report = FaissValidationReport(
        total_documents=len(documents),
        total_chunks=len(chunks),
        total_embeddings=emb_count,
        index_vector_count=index_count,
        index_matches_embeddings=index_matches,
        load_save_ok=load_save_ok,
        all_searches_non_empty=all_non_empty,
        metadata_consistent=meta_ok_all,
        hnsw_config={
            "m": settings.default_hnsw_m,
            "ef_construction": settings.default_hnsw_ef_construction,
            "ef_search": settings.default_hnsw_ef_search,
        },
        embedding_model=settings.embedding_model_name,
        queries=query_reports,
    )

    if args.save_report:
        out = Path("data/processed/faiss_report.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        base = asdict(report)
        queries_serialized = base.pop("queries")
        payload = {"report": {**base, "queries": queries_serialized}}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved report to {out}")

    return 0 if index_matches and all_non_empty and meta_ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
