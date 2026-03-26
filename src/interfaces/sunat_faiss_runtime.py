"""Montar retrieve → rerank → generación reutilizando un índice FAISS guardado en disco.

Evita re-ejecutar ingesta, chunking ni re-embedder del corpus completo: solo carga vectores
y metadatos, reconstruye BM25 en memoria desde los textos del JSON de metadatos, y usa el
mismo modelo de embeddings solo para la consulta y las citas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dataclasses import replace

from src.application.use_cases.build_bm25_index import BuildBm25IndexUseCase
from src.application.use_cases.cite_answer import CiteAnswerUseCase
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.rerank_context import RerankContextUseCase
from src.application.use_cases.retrieve_context import RetrieveContextUseCase
from src.config import get_settings
from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval import HybridRetrieverConfig
from src.infrastructure.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from src.infrastructure.llms.qwen_generator import QwenGenerator
from src.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from src.infrastructure.vectorstores.faiss_hnsw_store import FAISSHNSWStore

logger = logging.getLogger(__name__)


def chunks_from_faiss_metadata_dir(faiss_dir: str | Path) -> list[Chunk]:
    """Reconstruye entidades Chunk desde ``faiss_hnsw_metadata.json`` (alineado al índice)."""
    faiss_dir = Path(faiss_dir)
    meta_path = faiss_dir / "faiss_hnsw_metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    out: list[Chunk] = []
    for m in payload.get("metadata", []):
        out.append(
            Chunk(
                chunk_id=m["chunk_id"],
                doc_id=m["doc_id"],
                text=m["text"],
                source=m["source"],
                page=m.get("page"),
            )
        )
    return out


def answer_from_saved_faiss(
    question: str,
    *,
    faiss_dir: str | Path,
    top_k: int = 5,
    hybrid_pool: int | None = None,
) -> Answer:
    """
    Ejecuta híbrido + rerank + Qwen + citas usando ``FAISSHNSWStore.load(faiss_dir)``.

    Requiere en ``faiss_dir``: ``faiss_hnsw.index`` y ``faiss_hnsw_metadata.json`` (pareja
    generada por ``FAISSHNSWStore.save``).
    """
    faiss_dir = Path(faiss_dir)
    settings = get_settings()
    pool = hybrid_pool if hybrid_pool is not None else max(settings.hybrid_final_top_k, top_k)

    store = FAISSHNSWStore.load(str(faiss_dir))
    chunks = chunks_from_faiss_metadata_dir(faiss_dir)
    if len(chunks) != store.vector_count:
        logger.warning(
            "Chunks en metadata (%s) != vectores en índice (%s); BM25 puede desalinearse.",
            len(chunks),
            store.vector_count,
        )

    bm25 = BuildBm25IndexUseCase().execute(chunks)
    base_h = RetrieveContextUseCase.default_hybrid_config()
    hcfg: HybridRetrieverConfig = replace(
        base_h,
        top_k_dense=max(base_h.top_k_dense, pool),
        top_k_sparse=max(base_h.top_k_sparse, pool),
        final_top_k=pool,
    )
    embedder = SentenceTransformerEmbeddings(model_name=settings.embedding_model_name)
    hybrid_engine = HybridRetriever(hcfg)
    retrieve = RetrieveContextUseCase(
        embedding_adapter=embedder,
        vector_store=store,
        bm25_retriever=bm25,
        hybrid_retriever=hybrid_engine,
    )
    rerank = RerankContextUseCase(reranker=CrossEncoderReranker())
    generate = GenerateAnswerUseCase(
        llm=QwenGenerator(),
        citation_use_case=CiteAnswerUseCase(embedder),
    )

    hybrid = retrieve.execute(question)
    reranked = rerank.execute(question, hybrid, top_k)
    return generate.execute(question, reranked)
