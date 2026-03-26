from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings decoupled from environment and host paths."""

    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    eval_data_dir: Path
    default_chunk_size: int
    default_chunk_overlap: int
    default_min_chunk_length: int
    default_top_k: int
    embedding_model_name: str
    default_hnsw_m: int
    default_hnsw_ef_construction: int
    default_hnsw_ef_search: int
    reranker_model_name: str
    reranker_batch_size: int
    generation_model_name: str
    generation_max_new_tokens: int
    generation_temperature: float
    generation_top_p: float
    generation_do_sample: bool
    vector_index_name: str
    hybrid_dense_weight: float
    hybrid_sparse_weight: float
    hybrid_top_k_dense: int
    hybrid_top_k_sparse: int
    hybrid_final_top_k: int
    hybrid_score_normalization: str
    hybrid_fusion_temperature: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build cached settings from environment variables."""
    root = Path(os.getenv("RAG_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    data_dir = root / "data"
    return Settings(
        project_root=root,
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        processed_data_dir=data_dir / "processed",
        eval_data_dir=data_dir / "eval",
        default_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
        default_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
        default_min_chunk_length=int(os.getenv("RAG_MIN_CHUNK_LENGTH", "120")),
        default_top_k=int(os.getenv("RAG_TOP_K", "5")),
        embedding_model_name=os.getenv("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        default_hnsw_m=int(os.getenv("RAG_HNSW_M", "24")),
        default_hnsw_ef_construction=int(os.getenv("RAG_HNSW_EF_CONSTRUCTION", "120")),
        default_hnsw_ef_search=int(os.getenv("RAG_HNSW_EF_SEARCH", "48")),
        reranker_model_name=os.getenv(
            "RAG_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
        ),
        reranker_batch_size=int(os.getenv("RAG_RERANKER_BATCH_SIZE", "16")),
        generation_model_name=os.getenv(
            "RAG_GENERATION_MODEL",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        generation_max_new_tokens=int(os.getenv("RAG_GENERATION_MAX_NEW_TOKENS", "512")),
        generation_temperature=float(os.getenv("RAG_GENERATION_TEMPERATURE", "0.3")),
        generation_top_p=float(os.getenv("RAG_GENERATION_TOP_P", "0.9")),
        generation_do_sample=os.getenv("RAG_GENERATION_DO_SAMPLE", "true").lower() in ("1", "true", "yes"),
        vector_index_name=os.getenv("RAG_VECTOR_INDEX", "faiss_hnsw"),
        hybrid_dense_weight=float(os.getenv("RAG_HYBRID_DENSE_WEIGHT", "0.45")),
        hybrid_sparse_weight=float(os.getenv("RAG_HYBRID_SPARSE_WEIGHT", "0.55")),
        hybrid_top_k_dense=int(os.getenv("RAG_HYBRID_TOP_K_DENSE", "20")),
        hybrid_top_k_sparse=int(os.getenv("RAG_HYBRID_TOP_K_SPARSE", "20")),
        hybrid_final_top_k=int(os.getenv("RAG_HYBRID_FINAL_TOP_K", "10")),
        hybrid_score_normalization=os.getenv("RAG_HYBRID_SCORE_NORM", "softmax"),
        hybrid_fusion_temperature=float(os.getenv("RAG_HYBRID_FUSION_TEMPERATURE", "1.0")),
    )
