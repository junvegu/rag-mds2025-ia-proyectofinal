import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.entities.document import Document
from src.infrastructure.loaders.url_dataset_loader import URLDatasetLoader

SUNAT_URLS: list[str] = [
    "https://www.gob.pe/12274-declarar-y-pagar-impuesto-anual-para-rentas-de-trabajo-4ta-y-5ta-categoria",
    "https://www.sunat.gob.pe/legislacion/superin/2025/000386-2025.pdf",
    "https://www.gob.pe/9511-opciones-de-pago-electronico-de-impuestos-a-la-sunat",
    "https://www.gob.pe/1202-superintendencia-nacional-de-aduanas-y-de-administracion-tributaria-calcular-el-impuesto-a-la-renta-de-primera-categoria",
    "https://www.gob.pe/109657-declarar-y-pagar-la-renta-2025-primera-categoria",
    "https://www.gob.pe/8248-calcular-el-impuesto-a-la-renta-de-segunda-categoria-para-venta-de-valores-mobiliarios-y-ganancias-en-fondos-mutuosy/o",
    "https://www.gob.pe/7319-calcular-el-impuesto-a-la-renta-de-quinta-categoria",
    "https://www.gob.pe/7318-superintendencia-nacional-de-aduanas-y-de-administracion-tributaria-calcular-el-impuesto-de-cuarta-categoriay/o",
    "https://emprender.sunat.gob.pe/declaracion-pagos/pagos/fraccionamiento-deudas",
    "https://emprender.sunat.gob.pe/simuladores/fraccionamiento",
    "https://renta.sunat.gob.pe/sites/default/files/inline-files/Cartilla%20Instrucciones%20Personas%20%281%29_0.pdf",
    "https://renta.sunat.gob.pe/sites/default/files/inline-files/cartilla%20Instrucciones%20Empresa_2_6.pdf",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionReport:
    total_sources: int
    successful_sources: int
    failed_sources: int
    total_documents: int
    total_characters: int
    average_characters_per_document: float
    min_characters_per_document: int
    max_characters_per_document: int
    near_empty_documents: int
    documents_by_source_type: dict[str, int]
    documents_by_url: dict[str, int]
    page_distribution: dict[str, int]
    estimated_chunks: dict[str, int]
    dataset_over_100_chunks: bool


def _estimate_chunks(total_chars: int, chunk_size: int, overlap: int) -> int:
    if total_chars <= 0:
        return 0
    step = max(1, chunk_size - overlap)
    return max(1, ((total_chars - chunk_size + step - 1) // step) + 1)


def _source_key(document: Document) -> str:
    return str(document.metadata.get("original_source", document.source))


def _collect_documents(loader: URLDatasetLoader, sources: list[str]) -> tuple[list[Document], dict[str, str]]:
    documents: list[Document] = []
    errors: dict[str, str] = {}
    for source in sources:
        try:
            source_docs = loader._load_one(source)  # intentional for per-source diagnostics
            documents.extend(source_docs)
            if not source_docs:
                errors[source] = "No documents extracted."
        except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
            logger.warning("Source failed: %s -> %s", source, exc)
            errors[source] = str(exc)
    return documents, errors


def _build_report(sources: list[str], documents: list[Document], errors: dict[str, str]) -> IngestionReport:
    documents_by_url: dict[str, int] = {source: 0 for source in sources}
    documents_by_source_type: dict[str, int] = {}
    page_distribution: dict[str, int] = {"with_page": 0, "without_page": 0}

    lengths = [len(doc.text) for doc in documents]
    for doc in documents:
        origin = _source_key(doc)
        documents_by_url[origin] = documents_by_url.get(origin, 0) + 1

        source_type = str(doc.metadata.get("source_type", "unknown"))
        documents_by_source_type[source_type] = documents_by_source_type.get(source_type, 0) + 1

        if doc.page is None:
            page_distribution["without_page"] += 1
        else:
            page_distribution["with_page"] += 1

    total_chars = sum(lengths)
    estimated = {
        "chunk_500_overlap_50": _estimate_chunks(total_chars, chunk_size=500, overlap=50),
        "chunk_800_overlap_100": _estimate_chunks(total_chars, chunk_size=800, overlap=100),
        "chunk_1000_overlap_100": _estimate_chunks(total_chars, chunk_size=1000, overlap=100),
    }

    return IngestionReport(
        total_sources=len(sources),
        successful_sources=len(sources) - len(errors),
        failed_sources=len(errors),
        total_documents=len(documents),
        total_characters=total_chars,
        average_characters_per_document=mean(lengths) if lengths else 0.0,
        min_characters_per_document=min(lengths) if lengths else 0,
        max_characters_per_document=max(lengths) if lengths else 0,
        near_empty_documents=sum(1 for size in lengths if size < 80),
        documents_by_source_type=documents_by_source_type,
        documents_by_url=documents_by_url,
        page_distribution=page_distribution,
        estimated_chunks=estimated,
        dataset_over_100_chunks=any(value > 100 for value in estimated.values()),
    )


def _print_report(report: IngestionReport, documents: list[Document], errors: dict[str, str]) -> None:
    print("=== SUNAT Ingestion Report ===")
    print(f"Total sources processed : {report.total_sources}")
    print(f"Successful sources      : {report.successful_sources}")
    print(f"Failed sources          : {report.failed_sources}")
    print(f"Total documents         : {report.total_documents}")
    print(f"Total characters        : {report.total_characters}")
    print(f"Average text length     : {report.average_characters_per_document:.2f}")
    print(f"Min/Max text length     : {report.min_characters_per_document}/{report.max_characters_per_document}")
    print(f"Near-empty documents    : {report.near_empty_documents}")

    print("\nDocuments by source type:")
    for key, value in sorted(report.documents_by_source_type.items()):
        print(f"- {key}: {value}")

    print("\nDocuments by URL/source:")
    for source, count in report.documents_by_url.items():
        print(f"- {source}: {count}")

    print("\nEstimated chunks:")
    for scenario, value in report.estimated_chunks.items():
        print(f"- {scenario}: {value}")
    print(f"Dataset likely >100 chunks: {report.dataset_over_100_chunks}")

    print("\nPage distribution:")
    for key, value in report.page_distribution.items():
        print(f"- {key}: {value}")

    if errors:
        print("\nWarnings / failed sources:")
        for source, message in errors.items():
            print(f"- {source}: {message}")

    print("\nTop 10 longest documents:")
    for doc in sorted(documents, key=lambda item: len(item.text), reverse=True)[:10]:
        print(f"- {doc.title} ({len(doc.text)} chars) [{_source_key(doc)}]")

    print("\nTop 10 shortest documents:")
    for doc in sorted(documents, key=lambda item: len(item.text))[:10]:
        print(f"- {doc.title} ({len(doc.text)} chars) [{_source_key(doc)}]")

    print("\nSample content (3 documents):")
    for doc in documents[:3]:
        sample = doc.text[:220].replace("\n", " ").strip()
        if len(doc.text) > 220:
            sample += "..."
        print(f"- {doc.title} | source={doc.source} | page={doc.page}")
        print(f"  {sample}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ingestion QA for SUNAT dataset sources.")
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save JSON report to data/processed/ingestion_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    loader = URLDatasetLoader()

    documents, errors = _collect_documents(loader=loader, sources=SUNAT_URLS)
    report = _build_report(sources=SUNAT_URLS, documents=documents, errors=errors)
    _print_report(report=report, documents=documents, errors=errors)

    if args.save_report:
        output_path = Path("data/processed/ingestion_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "report": asdict(report),
            "errors": errors,
            "sample_documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "page": doc.page,
                    "text_length": len(doc.text),
                }
                for doc in documents[:10]
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
