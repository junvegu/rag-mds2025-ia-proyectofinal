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
    generation_model_name: str
    vector_index_name: str


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
        reranker_model_name=os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base"),
        generation_model_name=os.getenv("RAG_GENERATION_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        vector_index_name=os.getenv("RAG_VECTOR_INDEX", "faiss_hnsw"),
    )
