from src.application.ports.chunker_port import ChunkerPort
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.infrastructure.loaders.text_utils import normalize_text


class RecursiveChunker(ChunkerPort):
    """Character-based chunker with overlap and minimum chunk filtering."""

    def __init__(self, min_chunk_length: int = 120) -> None:
        self._min_chunk_length = min_chunk_length

    def chunk(
        self,
        document: Document,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_length: int | None = None,
    ) -> list[Chunk]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        effective_min_chunk = self._min_chunk_length if min_chunk_length is None else min_chunk_length
        text = normalize_text(document.text)
        if not text:
            return []

        windows = self._build_windows(text_length=len(text), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks: list[Chunk] = []
        for idx, (start_char, end_char) in enumerate(windows):
            chunk_text = normalize_text(text[start_char:end_char])
            if len(chunk_text) < effective_min_chunk:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-{start_char}-{end_char}",
                    doc_id=document.doc_id,
                    text=chunk_text,
                    source=document.source,
                    page=document.page,
                    chunk_index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    overlap_left=0 if idx == 0 else chunk_overlap,
                    overlap_right=0 if idx == len(windows) - 1 else chunk_overlap,
                    metadata={
                        **document.metadata,
                        "source": document.source,
                        "page": document.page,
                        "chunk_index": idx,
                        "total_chunks_candidate": len(windows),
                    },
                )
            )
        return chunks

    @staticmethod
    def _build_windows(text_length: int, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
        step = chunk_size - chunk_overlap
        windows: list[tuple[int, int]] = []
        start = 0
        while start < text_length:
            end = min(start + chunk_size, text_length)
            windows.append((start, end))
            if end == text_length:
                break
            start += step
        return windows
