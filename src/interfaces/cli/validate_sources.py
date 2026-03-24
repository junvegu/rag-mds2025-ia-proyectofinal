import argparse
import sys
from statistics import mean

from src.application.use_cases.load_documents import LoadDocumentsUseCase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate SUNAT sources (PDF + URL) before chunking."
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Path to one SUNAT PDF file.",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="One SUNAT URL to validate web loading.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of characters to print as content sample.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    use_case = LoadDocumentsUseCase()

    documents = use_case.execute([args.pdf_path, args.url])
    print("=== SUNAT Source Validation ===")
    print(f"Total documentos cargados: {len(documents)}")

    if not documents:
        print("No se cargaron documentos. Verifica ruta PDF, URL y dependencias.")
        return 1

    text_lengths = [len(doc.text) for doc in documents if doc.text]
    average_length = mean(text_lengths) if text_lengths else 0.0
    print(f"Longitud promedio de texto: {average_length:.2f} caracteres")

    first_document = documents[0]
    sample = first_document.text[: max(0, args.sample_size)].replace("\n", " ").strip()
    print("\nEjemplo de contenido:")
    print(sample if sample else "[sin contenido]")

    print("\nMetadata ejemplo:")
    print(
        f"doc_id={first_document.doc_id} | title={first_document.title} | "
        f"source={first_document.source} | page={first_document.page}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
