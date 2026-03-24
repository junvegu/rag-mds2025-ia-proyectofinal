from src.application.ports.chunker_port import ChunkerPort
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.infrastructure.loaders.text_utils import normalize_text


class TextChunker(ChunkerPort):
    """Deterministic character-based chunking with overlap."""

    def chunk(self, document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        text = normalize_text(document.content)
        if not text:
            return []

        step = chunk_size - chunk_overlap
        windows: list[tuple[int, int]] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            windows.append((start, end))
            if end == text_length:
                break
            start += step

        total_chunks = len(windows)
        chunks: list[Chunk] = []
        for idx, (start_char, end_char) in enumerate(windows):
            left_overlap = 0 if idx == 0 else min(chunk_overlap, start_char)
            right_overlap = 0 if idx == total_chunks - 1 else chunk_overlap
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}-{idx}",
                    doc_id=document.document_id,
                    text=text[start_char:end_char],
                    source=document.source,
                    page=document.page,
                    chunk_index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    overlap_left=left_overlap,
                    overlap_right=right_overlap,
                    metadata={
                        "source_file": document.metadata.get("source_file", ""),
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                    },
                )
            )
        return chunks
