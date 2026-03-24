import argparse
import sys
from pathlib import Path

from src.application.use_cases.process_documents_use_case import ProcessDocumentsUseCase
from src.config import get_settings
from src.infrastructure.chunking.text_chunker import TextChunker
from src.infrastructure.document_loaders.text_file_document_loader import TextFileDocumentLoader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local document loading + chunking flow.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Directory with .txt files. Defaults to configured data/raw path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size override. Defaults to config value.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap override. Defaults to config value.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of chunk examples to print.",
    )
    parser.add_argument(
        "--create-sample-if-empty",
        action="store_true",
        help="Create a sample .txt in source-dir if no files are found.",
    )
    return parser


def _create_sample_text_file(source_dir: Path) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    sample_path = source_dir / "sample_document.txt"
    if not sample_path.exists():
        sample_path.write_text(
            (
                "RAG systems combine retrieval and generation with grounded context.\n"
                "This sample file helps validate local loader and chunking flow.\n"
                "Next milestones will add embeddings, vector store, and generation.\n"
            ),
            encoding="utf-8",
        )
    return sample_path


def main() -> int:
    settings = get_settings()
    args = _build_parser().parse_args()

    source_dir = args.source_dir or str(settings.raw_data_dir)
    source_path = Path(source_dir)
    chunk_size = args.chunk_size or settings.default_chunk_size
    chunk_overlap = args.chunk_overlap or settings.default_chunk_overlap

    use_case = ProcessDocumentsUseCase(
        loader=TextFileDocumentLoader(),
        chunker=TextChunker(),
    )

    try:
        result = use_case.execute(
            source_dir=source_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print("=== RAG Document Processing ===")
    print(f"source_dir      : {source_dir}")
    print(f"chunk_size      : {chunk_size}")
    print(f"chunk_overlap   : {chunk_overlap}")
    print(f"total_documents : {len(result.documents)}")
    print(f"total_chunks    : {len(result.chunks)}")

    if not result.documents:
        if args.create_sample_if_empty:
            sample_path = _create_sample_text_file(source_path)
            print(f"No .txt documents found. Created sample file: {sample_path}")
            result = use_case.execute(
                source_dir=source_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            print("No .txt documents found. Add files to data/raw/ and rerun.")
            print("Tip: rerun with --create-sample-if-empty to auto-generate one sample file.")
            return 0

    if not result.documents:
        print("No .txt documents found. Add files to data/raw/ and rerun.")
        return 0

    print("\nSample chunks:")
    for chunk in result.chunks[: max(0, args.preview)]:
        source_file = chunk.metadata.get("source_file", "unknown")
        total_chunks = chunk.metadata.get("total_chunks", "?")
        preview_text = chunk.text.replace("\n", " ").strip()
        if len(preview_text) > 140:
            preview_text = preview_text[:140] + "..."
        print(
            f"- [{source_file}] chunk {chunk.metadata.get('chunk_index', chunk.chunk_index)}"
            f"/{total_chunks} chars={chunk.start_char}:{chunk.end_char}"
        )
        print(f"  {preview_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
