import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.use_cases.chunk_documents import ChunkDocumentsUseCase
from src.application.use_cases.load_documents import LoadDocumentsUseCase
from src.config import get_settings
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker
from src.infrastructure.loaders.text_utils import normalize_text

logger = logging.getLogger(__name__)

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


@dataclass(slots=True)
class ChunkingReport:
    input_documents: int
    generated_chunks: int
    avg_chunks_per_document: float
    avg_chunk_length: float
    min_chunk_length: int
    max_chunk_length: int
    discarded_short_chunks: int
    chunk_distribution_by_source: dict[str, int]
    chunk_distribution_by_page: dict[str, int]
    config: dict[str, int]
    assessment: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate chunking quality for SUNAT dataset.")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report in data/processed/chunking_report.json")
    return parser.parse_args()


def _estimate_discarded_chunks(
    documents: list[Document],
    chunks: list[Chunk],
    chunker: RecursiveChunker,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    generated_by_doc = Counter(chunk.doc_id for chunk in chunks)
    discarded = 0
    for doc in documents:
        normalized = normalize_text(doc.text)
        if not normalized:
            continue
        candidate_windows = chunker._build_windows(len(normalized), chunk_size, chunk_overlap)
        discarded += max(0, len(candidate_windows) - generated_by_doc.get(doc.doc_id, 0))
    return discarded


def _assess_configuration(report: ChunkingReport) -> str:
    if report.generated_chunks < 100 or report.avg_chunk_length < 220:
        return "chunking insuficiente"
    if report.generated_chunks <= 1200 and 250 <= report.avg_chunk_length <= 900:
        return "chunking recomendado"
    return "chunking aceptable"


def _print_examples(chunks: list[Chunk]) -> None:
    print("\n=== Ejemplos de chunks (3) ===")
    for chunk in chunks[:3]:
        sample = chunk.text[:220].replace("\n", " ").strip()
        if len(chunk.text) > 220:
            sample += "..."
        print(
            f"- chunk_id={chunk.chunk_id} | doc_id={chunk.doc_id} | "
            f"source={chunk.source} | page={chunk.page} | range={chunk.start_char}:{chunk.end_char}"
        )
        print(f"  {sample}")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    chunk_size = settings.default_chunk_size
    chunk_overlap = settings.default_chunk_overlap
    min_chunk_length = settings.default_min_chunk_length

    documents = LoadDocumentsUseCase().execute(SUNAT_URLS)
    chunker = RecursiveChunker(min_chunk_length=min_chunk_length)
    chunks = ChunkDocumentsUseCase(chunker=chunker).execute(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_length=min_chunk_length,
    )

    chunk_lengths = [len(chunk.text) for chunk in chunks]
    by_source = Counter(chunk.source for chunk in chunks)
    by_page = Counter("with_page" if chunk.page is not None else "without_page" for chunk in chunks)
    discarded = _estimate_discarded_chunks(
        documents=documents,
        chunks=chunks,
        chunker=chunker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    report = ChunkingReport(
        input_documents=len(documents),
        generated_chunks=len(chunks),
        avg_chunks_per_document=(len(chunks) / len(documents)) if documents else 0.0,
        avg_chunk_length=mean(chunk_lengths) if chunk_lengths else 0.0,
        min_chunk_length=min(chunk_lengths) if chunk_lengths else 0,
        max_chunk_length=max(chunk_lengths) if chunk_lengths else 0,
        discarded_short_chunks=discarded,
        chunk_distribution_by_source=dict(by_source),
        chunk_distribution_by_page=dict(by_page),
        config={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_chunk_length": min_chunk_length,
        },
        assessment="pending",
    )
    report.assessment = _assess_configuration(report)

    print("=== Chunking Validation Report ===")
    print(f"Input documents            : {report.input_documents}")
    print(f"Generated chunks           : {report.generated_chunks}")
    print(f"Average chunks/document    : {report.avg_chunks_per_document:.2f}")
    print(f"Average chunk length       : {report.avg_chunk_length:.2f}")
    print(f"Shortest / longest chunk   : {report.min_chunk_length} / {report.max_chunk_length}")
    print(f"Discarded short chunks     : {report.discarded_short_chunks}")
    print(f"Config                     : {report.config}")

    print("\nChunk distribution by source:")
    for source, count in report.chunk_distribution_by_source.items():
        print(f"- {source}: {count}")

    print("\nChunk distribution by page:")
    for key, count in report.chunk_distribution_by_page.items():
        print(f"- {key}: {count}")

    _print_examples(chunks)

    print("\nAssessment for next phase:")
    print("- Retrieval: good if average chunk length remains in medium range.")
    print("- Reranking: good if chunks retain enough local context.")
    print("- Colab speed: good if chunk count is not excessively high.")
    print(f"\nFinal conclusion: {report.assessment}")

    if args.save_report:
        output_path = Path("data/processed/chunking_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"report": asdict(report)}
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
