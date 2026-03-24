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
