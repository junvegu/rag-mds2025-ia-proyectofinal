import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.use_cases.chunk_documents import ChunkDocumentsUseCase
from src.application.use_cases.load_documents import LoadDocumentsUseCase
from src.config import get_settings
from src.domain.entities.chunk import Chunk
from src.infrastructure.chunking.recursive_chunker import RecursiveChunker
from src.infrastructure.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings

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
class EmbeddingExample:
    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    text_preview: str
    embedding_shape: int


@dataclass(slots=True)
class EmbeddingReport:
    total_documents: int
    total_chunks: int
    total_embeddings: int
    embedding_dimension: int
    generation_seconds: float
    avg_seconds_per_chunk: float
    model_name: str
    batch_size: int
    error_count: int
    all_chunks_embedded: bool
    consistent_dimension: bool
    has_empty_or_null_embeddings: bool
    examples: list[EmbeddingExample]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate embedding generation over SUNAT chunks.")
    parser.add_argument("--save-report", action="store_true", help="Save report JSON to data/processed/embeddings_report.json")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    return parser.parse_args()


def _embed_chunks_with_error_tracking(
    chunks: list[Chunk],
    model_name: str,
    batch_size: int,
) -> tuple[list[tuple[Chunk, list[float]]], int]:
    adapter = SentenceTransformerEmbeddings(model_name=model_name, batch_size=batch_size)
    pairs: list[tuple[Chunk, list[float]]] = []
    errors = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        try:
            vectors = adapter.embed_texts(texts)
        except Exception as exc:  # noqa: BLE001
            errors += len(batch)
            logger.warning("Embedding batch failed at offset %s: %s", start, exc)
            continue

        if len(vectors) != len(batch):
            errors += len(batch)
            logger.warning("Embedding batch size mismatch at offset %s.", start)
            continue

        for chunk, vector in zip(batch, vectors):
            if not vector:
                errors += 1
                continue
            pairs.append((chunk, vector))
    return pairs, errors


def _validate_embeddings(pairs: list[tuple[Chunk, list[float]]], total_chunks: int) -> tuple[bool, bool, bool, int]:
    if not pairs:
        return False, True, True, 0

    dimensions = {len(vector) for _, vector in pairs}
    has_empty = any(len(vector) == 0 for _, vector in pairs)
    all_chunks_embedded = len(pairs) == total_chunks
    consistent_dimension = len(dimensions) == 1
    dimension = next(iter(dimensions))
    return all_chunks_embedded, consistent_dimension, has_empty, dimension


def _build_examples(pairs: list[tuple[Chunk, list[float]]]) -> list[EmbeddingExample]:
    examples: list[EmbeddingExample] = []
    for chunk, vector in pairs[:3]:
        preview = chunk.text[:160].replace("\n", " ").strip()
        if len(chunk.text) > 160:
            preview += "..."
        examples.append(
            EmbeddingExample(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                page=chunk.page,
                text_preview=preview,
                embedding_shape=len(vector),
            )
        )
    return examples


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()

    documents = LoadDocumentsUseCase().execute(SUNAT_URLS)
    chunker = RecursiveChunker(min_chunk_length=settings.default_min_chunk_length)
    chunks = ChunkDocumentsUseCase(chunker=chunker).execute(
        documents=documents,
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
        min_chunk_length=settings.default_min_chunk_length,
    )

    start = time.perf_counter()
    pairs, error_count = _embed_chunks_with_error_tracking(
        chunks=chunks,
        model_name=settings.embedding_model_name,
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - start

    all_chunks_embedded, consistent_dimension, has_empty, dimension = _validate_embeddings(
        pairs=pairs,
        total_chunks=len(chunks),
    )
    report = EmbeddingReport(
        total_documents=len(documents),
        total_chunks=len(chunks),
        total_embeddings=len(pairs),
        embedding_dimension=dimension,
        generation_seconds=elapsed,
        avg_seconds_per_chunk=(elapsed / len(pairs)) if pairs else 0.0,
        model_name=settings.embedding_model_name,
        batch_size=args.batch_size,
        error_count=error_count,
        all_chunks_embedded=all_chunks_embedded,
        consistent_dimension=consistent_dimension,
        has_empty_or_null_embeddings=has_empty,
        examples=_build_examples(pairs),
    )

    print("=== Embeddings Validation Report ===")
    print(f"Total documentos          : {report.total_documents}")
    print(f"Total chunks              : {report.total_chunks}")
    print(f"Total embeddings          : {report.total_embeddings}")
    print(f"Embedding dimension       : {report.embedding_dimension}")
    print(f"Tiempo total (s)          : {report.generation_seconds:.4f}")
    print(f"Tiempo promedio/chunk (s) : {report.avg_seconds_per_chunk:.6f}")
    print(f"Modelo                    : {report.model_name}")
    print(f"Batch size                : {report.batch_size}")
    print(f"Errores                   : {report.error_count}")

    print("\nValidaciones:")
    print(f"- Todos los chunks embebidos       : {report.all_chunks_embedded}")
    print(f"- Dimension consistente             : {report.consistent_dimension}")
    print(f"- Embeddings vacios o nulos         : {report.has_empty_or_null_embeddings}")

    print("\nEjemplos (3):")
    for example in report.examples:
        print(
            f"- chunk_id={example.chunk_id} | doc_id={example.doc_id} | source={example.source} "
            f"| page={example.page} | dim={example.embedding_shape}"
        )
        print(f"  {example.text_preview}")

    if args.save_report:
        output_path = Path("data/processed/embeddings_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"report": asdict(report)}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
