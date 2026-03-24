import logging
from collections import Counter, defaultdict
from statistics import mean

from src.application.use_cases.load_documents import LoadDocumentsUseCase
from src.domain.entities.document import Document

SUNAT_SOURCES: list[str] = [
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


def _print_examples(documents: list[Document], count: int = 3) -> None:
    print("\n=== Ejemplos de documentos (3) ===")
    for idx, doc in enumerate(documents[:count], start=1):
        sample = doc.text[:220].replace("\n", " ").strip()
        if len(doc.text) > 220:
            sample += "..."
        print(f"{idx}. title={doc.title!r} | source={doc.source} | page={doc.page}")
        print(f"   text_sample={sample}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    use_case = LoadDocumentsUseCase()

    documents = use_case.execute(SUNAT_SOURCES)
    loaded_by_original = Counter(doc.metadata.get("original_source", doc.source) for doc in documents)
    source_type_distribution = Counter(doc.metadata.get("source_type", "unknown") for doc in documents)
    avg_length = mean([len(doc.text) for doc in documents]) if documents else 0.0

    print("=== Validacion de Dataset SUNAT ===")
    print(f"Fuentes declaradas: {len(SUNAT_SOURCES)}")
    print(f"Total Document cargados: {len(documents)}")
    print(f"Longitud promedio de texto: {avg_length:.2f} caracteres")

    print("\n=== Distribucion por tipo de fuente ===")
    for key, value in sorted(source_type_distribution.items()):
        print(f"- {key}: {value}")

    print("\n=== Documentos por URL/fuente ===")
    for source in SUNAT_SOURCES:
        count = loaded_by_original.get(source, 0)
        status = "OK" if count > 0 else "WARNING: sin documentos"
        print(f"- {source} -> {count} ({status})")

    pages_per_source: dict[str, int] = defaultdict(int)
    for doc in documents:
        origin = doc.metadata.get("original_source", doc.source)
        pages_per_source[origin] += 1

    print("\n=== Paginas/Documentos por fuente ===")
    for source in SUNAT_SOURCES:
        print(f"- {source}: {pages_per_source.get(source, 0)}")

    _print_examples(documents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
