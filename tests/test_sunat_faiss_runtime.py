"""Metadata → Chunk reconstruction for demo runtime."""

from __future__ import annotations

import json
from pathlib import Path

from src.interfaces.sunat_faiss_runtime import chunks_from_faiss_metadata_dir


def test_chunks_from_faiss_metadata_dir(tmp_path: Path) -> None:
    d = tmp_path / "idx"
    d.mkdir()
    payload = {
        "dimension": 384,
        "m": 24,
        "ef_construction": 120,
        "ef_search": 48,
        "metadata": [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "text": "texto de prueba",
                "source": "https://example.com",
                "page": 3,
            }
        ],
    }
    (d / "faiss_hnsw_metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    chunks = chunks_from_faiss_metadata_dir(d)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].page == 3
