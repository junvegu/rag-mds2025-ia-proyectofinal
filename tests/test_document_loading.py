from dataclasses import dataclass

import pytest

from src.application.use_cases.load_documents import LoadDocumentsUseCase
from src.domain.entities.document import Document
from src.infrastructure.loaders.text_utils import normalize_text
from src.infrastructure.loaders.url_dataset_loader import URLDatasetLoader


def _sample_document(idx: int = 1, page: int | None = None) -> Document:
    return Document(
        doc_id=f"doc-{idx}",
        title=f"Title {idx}",
        text=f"Contenido de prueba {idx}",
        source=f"https://source/{idx}",
        page=page,
        metadata={"source_type": "url_html", "original_source": f"https://source/{idx}"},
    )


@dataclass
class _FakeDatasetLoader:
    def load(self, sources: list[str]) -> list[Document]:
        return [_sample_document(1), _sample_document(2)]


def test_load_documents_use_case_returns_list() -> None:
    use_case = LoadDocumentsUseCase(dataset_loader=_FakeDatasetLoader())  # type: ignore[arg-type]
    result = use_case.execute(["https://source/1"])
    assert isinstance(result, list)


def test_load_documents_use_case_returns_document_instances() -> None:
    use_case = LoadDocumentsUseCase(dataset_loader=_FakeDatasetLoader())  # type: ignore[arg-type]
    result = use_case.execute(["https://source/1"])
    assert result
    assert all(isinstance(item, Document) for item in result)


def test_loaded_documents_have_core_fields() -> None:
    use_case = LoadDocumentsUseCase(dataset_loader=_FakeDatasetLoader())  # type: ignore[arg-type]
    documents = use_case.execute(["https://source/1"])
    assert any(doc.title.strip() for doc in documents)
    assert any(doc.text.strip() for doc in documents)
    assert any(doc.source.strip() for doc in documents)


def test_pdf_documents_preserve_page_when_available() -> None:
    loader = URLDatasetLoader()

    class _FakePDFLoader:
        def load_from_url(self, url: str) -> list[Document]:
            return [_sample_document(1, page=3)]

        def load(self, path: str) -> list[Document]:
            return [_sample_document(2, page=1)]

    loader.pdf_loader = _FakePDFLoader()  # type: ignore[assignment]
    # URL termina en .pdf → _detect_remote_kind devuelve "pdf" sin HEAD
    docs = loader._load_one("https://example.com/sunat.pdf")
    assert docs
    assert docs[0].page == 3


def test_normalize_text_removes_repeated_spaces() -> None:
    raw = "Texto    con   espacios\r\n\r\n\r\nextra"
    normalized = normalize_text(raw)
    assert "    " not in normalized
    assert "\r" not in normalized


def test_loader_continues_when_one_source_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = ["ok-source", "bad-source", "ok-source-2"]

    def _fake_load_one(self: URLDatasetLoader, source: str) -> list[Document]:
        if source == "bad-source":
            raise ValueError("boom")
        return [_sample_document(99)]

    monkeypatch.setattr(URLDatasetLoader, "_load_one", _fake_load_one)
    loader = URLDatasetLoader()
    docs = loader.load(sources)
    assert len(docs) == 2


@pytest.mark.integration
def test_integration_real_sunat_sources_smoke() -> None:
    pytest.importorskip("requests")
    pytest.importorskip("bs4")
    pytest.importorskip("pypdf")

    sources = [
        "https://www.gob.pe/12274-declarar-y-pagar-impuesto-anual-para-rentas-de-trabajo-4ta-y-5ta-categoria",
        "https://www.sunat.gob.pe/legislacion/superin/2025/000386-2025.pdf",
    ]
    docs = URLDatasetLoader().load(sources)
    assert isinstance(docs, list)
