"""Interface adapters for notebook and local execution."""

from .container import build_pipeline
from .rag_pipeline import RAGPipeline
from .sunat_faiss_runtime import answer_from_saved_faiss, chunks_from_faiss_metadata_dir

__all__ = [
    "RAGPipeline",
    "answer_from_saved_faiss",
    "build_pipeline",
    "chunks_from_faiss_metadata_dir",
]
