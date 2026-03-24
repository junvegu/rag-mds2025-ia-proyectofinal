"""Document ingestion adapters."""

from .local_document_loader_stub import LocalDocumentLoaderStub
from .pdf_loader import PDFLoader
from .txt_loader import TXTLoader
from .url_dataset_loader import URLDatasetLoader
from .web_loader import WebLoader

__all__ = [
    "LocalDocumentLoaderStub",
    "TXTLoader",
    "PDFLoader",
    "WebLoader",
    "URLDatasetLoader",
]
