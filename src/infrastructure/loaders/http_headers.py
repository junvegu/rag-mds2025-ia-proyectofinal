"""Cabeceras HTTP reutilizables para ingesta (reduce 403/418 frente a peticiones “de bot”)."""

from __future__ import annotations

import os


def browser_like_headers(*, for_pdf: bool = False) -> dict[str, str]:
    """
    User-Agent y Accept razonables para sitios públicos (gob.pe, SUNAT).

    Personaliza con la variable de entorno ``RAG_HTTP_USER_AGENT`` si hace falta.
    """
    user_agent = os.environ.get(
        "RAG_HTTP_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    )
    if for_pdf:
        accept = "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8"
    else:
        accept = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    return {
        "User-Agent": user_agent,
        "Accept": accept,
        "Accept-Language": "es-PE,es;q=0.9,en;q=0.7",
    }
