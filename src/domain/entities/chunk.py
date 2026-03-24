from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    """Document fragment produced by a chunking strategy."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    overlap_left: int | None = None
    overlap_right: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
