"""Unit tests for BM25 + hybrid fusion."""

from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval import HybridRetrieverConfig, RetrievedChunk
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever, Bm25Hit, tokenize_spanish
from src.infrastructure.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_config_sunat_defaults() -> None:
    cfg = HybridRetrieverConfig()
    assert cfg.dense_weight == 0.45
    assert cfg.sparse_weight == 0.55
    assert cfg.score_normalization == "softmax"
    assert cfg.fusion_temperature == 1.0


def test_softmax_prefers_higher_raw_score() -> None:
    from src.infrastructure.retrieval.hybrid_retriever import _softmax_norm

    s = _softmax_norm({"a": 1.0, "b": 2.0, "c": 0.5}, temperature=1.0)
    assert s["b"] > s["a"] > s["c"]


def test_tokenize_spanish_basic() -> None:
    toks = tokenize_spanish("Renta de quinta categoría y SUNAT 2025")
    assert "quinta" in toks
    assert "categoría" in toks
    assert "2025" in toks


def test_bm25_search_order() -> None:
    chunks = [
        Chunk(
            chunk_id="a",
            doc_id="d1",
            text="impuesto quinta categoría trabajo dependiente",
            source="s1",
            page=1,
        ),
        Chunk(
            chunk_id="b",
            doc_id="d1",
            text="primera categoría alquiler de predios",
            source="s1",
            page=2,
        ),
    ]
    r = BM25Retriever()
    r.fit(chunks)
    hits = r.search("quinta categoría impuesto", top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "a"
    assert hits[0].score >= hits[1].score


def test_hybrid_fuse_dedup_and_scores() -> None:
    ch = Chunk(chunk_id="x", doc_id="d", text="hello world", source="u", page=None)
    dense = [
        RetrievedChunk(chunk=ch, dense_score=0.9),
    ]
    sparse = [
        Bm25Hit(chunk_id="x", doc_id="d", source="u", page=None, text="hello world", score=10.0),
        Bm25Hit(chunk_id="y", doc_id="d", source="u", page=None, text="other", score=5.0),
    ]
    cfg = HybridRetrieverConfig(
        dense_weight=0.5,
        sparse_weight=0.5,
        top_k_dense=5,
        top_k_sparse=5,
        final_top_k=5,
        score_normalization="minmax",
    )
    hy = HybridRetriever(cfg)
    out = hy.fuse(dense, sparse)
    ids = [r.chunk_id for r in out]
    assert ids.count("x") == 1
    assert out[0].chunk_id == "x"
    assert out[0].dense_score == 0.9
    assert out[0].sparse_score == 10.0
    assert out[0].hybrid_score > 0
