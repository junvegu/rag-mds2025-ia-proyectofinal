"""CiteAnswerUseCase y utilidades de sentence_citation."""

from __future__ import annotations

import numpy as np

from src.application.ports.embedding_port import EmbeddingPort
from src.application.use_cases.cite_answer import CiteAnswerUseCase
from src.domain.entities.retrieval import RerankedChunkResult
from src.infrastructure.citation.sentence_citation import (
    cosine_similarity_matrix,
    grounding_label_from_score,
    hallucination_flag_from_grounding,
    split_answer_sentences,
)


def _chunk(cid: str, text: str) -> RerankedChunkResult:
    return RerankedChunkResult(
        chunk_id=cid,
        doc_id="d",
        source="https://example.com/a",
        page=1,
        text=text,
        dense_score=0.1,
        sparse_score=0.2,
        hybrid_score=0.3,
        rerank_score=1.0,
        rerank_position=1,
    )


class _DeterministicEmbedder(EmbeddingPort):
    """Vectores fijos por índice de texto en el batch (solo para tests)."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.last_batch: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.last_batch = list(texts)
        out: list[list[float]] = []
        for i, _ in enumerate(texts):
            v = np.zeros(self.dim, dtype=np.float64)
            v[i % self.dim] = 1.0
            out.append(v.tolist())
        return out

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


def test_split_answer_sentences_basic() -> None:
    s = split_answer_sentences("Hola mundo. Segunda oración! ¿Tercera?")
    assert len(s) == 3
    assert s[0].startswith("Hola")


def test_grounding_thresholds_and_flag() -> None:
    assert grounding_label_from_score(0.8) == "well_grounded"
    assert grounding_label_from_score(0.6) == "partially_grounded"
    assert grounding_label_from_score(0.4) == "possible_hallucination"
    assert hallucination_flag_from_grounding(0.5) is True
    assert hallucination_flag_from_grounding(0.55) is False


def test_cosine_similarity_matrix_orthogonal() -> None:
    s = np.eye(3, 4)
    c = np.eye(2, 4)
    m = cosine_similarity_matrix(s, c)
    assert m.shape == (3, 2)


def test_cite_answer_single_batch_embedding_order() -> None:
    chunks = [_chunk("c1", "alpha text"), _chunk("c2", "beta text")]

    class _Emb(EmbeddingPort):
        def __init__(self) -> None:
            self.last_batch: list[str] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.last_batch = list(texts)
            out: list[list[float]] = []
            for t in texts:
                if "alpha" in t.lower():
                    out.append([1.0, 0.0])
                elif "beta" in t.lower():
                    out.append([0.0, 1.0])
                else:
                    out.append([0.5, 0.5])
            return out

        def embed_query(self, query: str) -> list[float]:
            return [0.0, 0.0]

    emb = _Emb()
    uc = CiteAnswerUseCase(emb)
    res = uc.execute("alpha match here. unrelated second.", chunks)
    assert len(emb.last_batch) == 4  # 2 chunks + 2 sentences
    assert emb.last_batch[0] == "alpha text"
    assert emb.last_batch[1] == "beta text"
    assert len(res.citations) == 2
    assert res.citations[0].chunk_id == "c1"
    assert 0.0 <= res.grounding_score <= 1.0
    assert isinstance(res.hallucination_flag, bool)


def test_cite_answer_empty_context() -> None:
    uc = CiteAnswerUseCase(_DeterministicEmbedder())
    r = uc.execute("Una oración.", [])
    assert r.citations == []
    assert r.grounding_score == 0.0
    assert r.hallucination_flag is True


def test_cite_answer_empty_answer() -> None:
    uc = CiteAnswerUseCase(_DeterministicEmbedder())
    r = uc.execute("   ", [_chunk("c1", "x")])
    assert r.citations == []
    assert r.grounding_score == 1.0
    assert r.hallucination_flag is False


def test_generate_answer_with_citations() -> None:
    from src.application.use_cases.generate_answer import GenerateAnswerUseCase
    from src.application.ports.llm_port import LLMPort

    class _LLM(LLMPort):
        def generate(self, *, system: str | None = None, user: str) -> str:
            return "Primera idea. Segunda idea."

    chunks = [_chunk("c1", "Primera idea sobre SUNAT."), _chunk("c2", "Otro tema.")]

    class _Emb(EmbeddingPort):
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            vecs: list[list[float]] = []
            for t in texts:
                if "Otro tema" in t:
                    vecs.append([0.0, 1.0])
                elif "Primera" in t:
                    vecs.append([1.0, 0.0])
                elif "Segunda" in t:
                    vecs.append([0.05, 0.95])
                else:
                    vecs.append([0.5, 0.5])
            return vecs

        def embed_query(self, query: str) -> list[float]:
            return [0.0, 0.0]

    cite = CiteAnswerUseCase(_Emb())
    ans = GenerateAnswerUseCase(llm=_LLM(), citation_use_case=cite).execute("¿Qué?", chunks)
    assert len(ans.citations) == 2
    assert ans.grounding_score is not None
    assert "grounding_label" in ans.metadata
    assert len(ans.metadata["sentence_citations"]) == 2
