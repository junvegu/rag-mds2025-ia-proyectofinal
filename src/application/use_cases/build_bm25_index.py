from dataclasses import dataclass

from src.domain.entities.chunk import Chunk
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever


@dataclass(slots=True)
class BuildBm25IndexUseCase:
    """Fit a BM25 index from the current chunk list."""

    def execute(self, chunks: list[Chunk]) -> BM25Retriever:
        retriever = BM25Retriever()
        retriever.fit(chunks)
        return retriever
