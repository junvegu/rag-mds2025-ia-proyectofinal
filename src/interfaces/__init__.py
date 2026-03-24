"""Interface adapters for notebook and local execution."""

from .container import build_pipeline
from .rag_pipeline import RAGPipeline

__all__ = ["RAGPipeline", "build_pipeline"]
