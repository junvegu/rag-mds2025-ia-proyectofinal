"""Unit tests for rerank port + use case (sin cargar CrossEncoder real)."""

from src.application.ports.reranker_port import RerankerPort
from src.application.use_cases.rerank_context import RerankContextUseCase
from src.domain.entities.retrieval import HybridChunkResult, RerankedChunkResult


class _FakeReranker(RerankerPort):
    """Ordena por hybrid_score descendente y asigna rerank_score = posición simulada."""

    def rerank(
        self,
        query: str,
        candidates: list[HybridChunkResult],
        top_k: int,
    ) -> list[RerankedChunkResult]:
        sorted_c = sorted(candidates, key=lambda c: c.hybrid_score, reverse=True)
        out: list[RerankedChunkResult] = []
        for pos, c in enumerate(sorted_c[:top_k], start=1):
            out.append(
                RerankedChunkResult.from_hybrid(
                    c,
                    rerank_score=float(pos),
                    rerank_position=pos,
                )
            )
        return out


def test_rerank_context_orders_and_limits() -> None:
    c1 = HybridChunkResult(
        chunk_id="a",
        doc_id="d",
        source="s",
        page=1,
        text="low",
        dense_score=0.1,
        sparse_score=0.2,
        hybrid_score=0.3,
    )
    c2 = HybridChunkResult(
        chunk_id="b",
        doc_id="d",
        source="s",
        page=2,
        text="high",
        dense_score=0.9,
        sparse_score=0.8,
        hybrid_score=0.95,
    )
    uc = RerankContextUseCase(reranker=_FakeReranker())
    out = uc.execute("pregunta", [c1, c2], top_k=1)
    assert len(out) == 1
    assert out[0].chunk_id == "b"
    assert out[0].rerank_position == 1
    assert out[0].hybrid_score == 0.95


def test_reranked_preserves_metadata() -> None:
    h = HybridChunkResult(
        chunk_id="x",
        doc_id="y",
        source="https://sunat.gob.pe/x",
        page=3,
        text="cuerpo",
        dense_score=0.5,
        sparse_score=1.0,
        hybrid_score=0.7,
    )
    r = RerankedChunkResult.from_hybrid(h, rerank_score=4.2, rerank_position=1)
    assert r.chunk_id == h.chunk_id and r.text == h.text and r.page == h.page
    assert r.hybrid_score == 0.7 and r.rerank_score == 4.2 and r.rerank_position == 1
