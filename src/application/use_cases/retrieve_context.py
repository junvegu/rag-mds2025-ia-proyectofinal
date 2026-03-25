from dataclasses import dataclass
from typing import cast

from src.application.ports.embedding_port import EmbeddingPort
from src.domain.entities.retrieval import HybridChunkResult, HybridRetrieverConfig, ScoreNormalizationMode
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever
from src.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from src.infrastructure.vectorstores.faiss_hnsw_store import FAISSHNSWStore


@dataclass(slots=True)
class RetrieveContextUseCase:
    """Dense + sparse retrieval and hybrid fusion for a natural-language query."""

    embedding_adapter: EmbeddingPort
    vector_store: FAISSHNSWStore
    bm25_retriever: BM25Retriever
    hybrid_retriever: HybridRetriever

    def execute(self, query: str) -> list[HybridChunkResult]:
        stripped = query.strip()
        if not stripped:
            raise ValueError("Query text cannot be empty.")

        cfg = self.hybrid_retriever.config
        query_vector = self.embedding_adapter.embed_query(stripped)
        dense_hits = self.vector_store.search(query_vector, cfg.top_k_dense)
        sparse_hits = self.bm25_retriever.search(stripped, cfg.top_k_sparse)
        return self.hybrid_retriever.fuse(dense_hits, sparse_hits)

    @staticmethod
    def default_hybrid_config() -> HybridRetrieverConfig:
        """Defaults aligned with Settings / env when used from scripts."""
        from src.config import get_settings

        s = get_settings()
        norm = s.hybrid_score_normalization.lower()
        if norm not in ("softmax", "minmax"):
            norm = "softmax"
        return HybridRetrieverConfig(
            dense_weight=s.hybrid_dense_weight,
            sparse_weight=s.hybrid_sparse_weight,
            top_k_dense=s.hybrid_top_k_dense,
            top_k_sparse=s.hybrid_top_k_sparse,
            final_top_k=s.hybrid_final_top_k,
            score_normalization=cast(ScoreNormalizationMode, norm),
            fusion_temperature=s.hybrid_fusion_temperature,
        )
