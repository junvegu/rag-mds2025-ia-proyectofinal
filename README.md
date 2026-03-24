# rag-colab-final

Esqueleto profesional para un sistema RAG académico con flujo recomendado:
**primero desarrollo local**, luego **migración simple a Google Colab**.

## Objetivo

Dejar una base sólida, pequeña y extensible para estas etapas:

- carga de documentos
- chunking con overlap
- embeddings
- FAISS
- retrieval
- reranker
- generación
- evaluación

## Arquitectura (pragmática y limpia)

```text
src/
  config/          -> configuración centralizada (paths, defaults, entorno)
  domain/          -> entidades y reglas de negocio
  application/     -> casos de uso y puertos (contratos)
  infrastructure/  -> implementaciones concretas de adaptadores
  interfaces/      -> punto de entrada para notebooks/scripts
```

Principios:

- El dominio no conoce librerías externas.
- Application define contratos; Infrastructure los implementa.
- Interfaces consume casos de uso, no detalles técnicos.
- Config central evita acoplar código a rutas absolutas o a Colab.

## Estructura final

```text
rag-colab-final/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── notebooks/
│   └── rag_final_demo.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── domain/
│   │   ├── entities/
│   │   └── services/
│   ├── application/
│   │   ├── ports/
│   │   └── use_cases/
│   ├── infrastructure/
│   │   ├── loaders/
│   │   ├── chunking/
│   │   ├── embeddings/
│   │   ├── vectorstores/
│   │   ├── retrieval/
│   │   ├── rerankers/
│   │   ├── llms/
│   │   └── evaluation/
│   └── interfaces/
│       ├── container.py
│       └── rag_pipeline.py
└── tests/
```

## Arranque local

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. (opcional) crear `.env` desde `.env.example`
5. `pytest`

Ejemplo rápido:

```python
from src.interfaces import build_pipeline

pipeline = build_pipeline()
result = pipeline.answer("¿Cuál es el objetivo del proyecto?")
print(result.text)
```

> Nota: en esta fase los adaptadores son **stubs** y no ejecutan embeddings, FAISS ni generación real.

## Flujo actual: carga + chunking

Ejecutar desde la raiz del proyecto:

- `python -m src.interfaces.cli.main`
- `python -m src.interfaces.cli.main --chunk-size 500 --chunk-overlap 80`
- `python -m src.interfaces.cli.main --create-sample-if-empty`

Salida esperada:

- total de documentos cargados desde `data/raw/`
- total de chunks generados
- muestra de chunks con metadata (`source_file`, `chunk_index`, `total_chunks`)

Defaults recomendados para el dataset SUNAT (111 documentos, mayoria PDF por pagina):

- `chunk_size=800`
- `chunk_overlap=100`
- `min_chunk_length=120`

Por que estos valores:

- `800` mantiene suficiente contexto normativo por chunk sin disparar demasiado el total de embeddings.
- `100` conserva continuidad entre chunks para no perder condiciones o incisos en cortes.
- `120` filtra fragmentos muy cortos/ruidosos frecuentes en pie de pagina o encabezados extraidos de PDF.

## Validacion de ingesta SUNAT

Script de validacion de fuentes web/PDF:

- `python -m src.interfaces.cli.load_sunat_dataset`
- `python3 scripts/test_sunat_ingestion.py --save-report`

El script imprime:

- total de documentos cargados
- distribucion por tipo de fuente
- numero de paginas/documentos por URL
- longitud promedio de texto
- ejemplo de 3 documentos cargados

Tests de ingesta:

- `pytest tests/test_document_loading.py`
- `pytest -m integration tests/test_document_loading.py` (requiere red + dependencias web/pdf)

## Bitacora de investigacion (paso a paso)

Esta seccion documenta decisiones tecnicas por fase, con evidencia en reportes versionados en `data/processed/`.

### Fase 1: Ingesta de fuentes SUNAT

Reporte base: `data/processed/ingestion_report.json`

Hallazgos principales:

- Fuentes evaluadas: `12`
- Fuentes exitosas: `5`
- Fuentes fallidas: `7` (varias URLs de `gob.pe` con `418`)
- Documentos cargados: `111`
- Caracteres totales: `337823`
- Distribucion por tipo:
  - `url_pdf`: `109`
  - `url_html`: `2`
- Distribucion por pagina:
  - `with_page`: `109`
  - `without_page`: `2`

Decision tomada:

- Mantener una estrategia robusta de ingesta tolerante a fallos por fuente para no detener el pipeline completo.
- Priorizar contenido PDF y preservar `page` para trazabilidad y futuras citas por oracion.
- Continuar a chunking, porque el volumen de texto ya era suficiente para retrieval.

### Fase 2: Chunking

Reporte base: `data/processed/chunking_report.json`

Configuracion evaluada:

- `chunk_size=800`
- `chunk_overlap=100`
- `min_chunk_length=120`

Resultados:

- Documentos de entrada: `111`
- Chunks generados: `514`
- Promedio chunks/documento: `4.63`
- Longitud promedio chunk: `735.40`
- Min/Max chunk: `126` / `800`
- Chunks descartados por cortos: `4`
- Evaluacion automatica: `chunking recomendado`

Decision tomada:

- Fijar defaults de chunking en `800/100/120`.
- Justificacion pragmatica:
  - buena cobertura de contexto legal por chunk,
  - solapamiento suficiente para continuidad semantica,
  - bajo ruido por descarte de fragmentos cortos.
- Resultado esperado para Colab: balance entre calidad y tiempo de procesamiento.

### Fase 3: Embeddings

Reporte base: `data/processed/embeddings_report.json`

Modelo evaluado:

- `intfloat/multilingual-e5-small`

Resultados:

- Documentos: `111`
- Chunks: `514`
- Embeddings generados: `514`
- Dimension: `384`
- Tiempo total: `171.26 s`
- Tiempo promedio por chunk: `0.333 s`
- Errores: `0`
- Consistencia:
  - `all_chunks_embedded=true`
  - `consistent_dimension=true`
  - `has_empty_or_null_embeddings=false`

Decision tomada:

- Dejar `intfloat/multilingual-e5-small` como default para este proyecto.
- Razon:
  - mejor equilibrio calidad/velocidad para espanol y entorno Colab,
  - costo computacional controlado para iteraciones academicas rapidas,
  - salida vectorial consistente (384-dim) lista para FAISS.

## Reportes y trazabilidad

Cada corrida de validacion guarda evidencia en `data/processed/`:

- `ingestion_report.json`: cobertura y calidad de fuentes cargadas
- `chunking_report.json`: calidad de segmentacion y configuracion de chunks
- `embeddings_report.json`: rendimiento y consistencia de vectores

Scripts asociados:

- `python3 scripts/test_sunat_ingestion.py --save-report`
- `python3 scripts/test_chunking.py --save-report`
- `python3 scripts/test_embeddings.py --save-report`

Sugerencia de trabajo para clase:

- mantener esta bitacora actualizada en cada fase (FAISS, retrieval, reranker, generacion),
- comparar decisiones con evidencia cuantitativa de los reportes,
- registrar cambios de parametros (modelo, chunking, batch size) y su impacto en tiempo/calidad.

## Migración futura a Colab (simple)

- Copiar/clonar el repositorio en Colab.
- Instalar `requirements.txt`.
- Ajustar variables por celda o `.env`:
  - `RAG_PROJECT_ROOT`
  - modelos por defecto (`RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL`, `RAG_GENERATION_MODEL`)
- Reutilizar `src/interfaces/rag_pipeline.py` como entrada estable.

## Estado

Bootstrap con contratos, entidades y configuración central.
La lógica completa de RAG queda para siguientes iteraciones.
