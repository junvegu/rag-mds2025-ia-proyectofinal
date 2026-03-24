"""Document loading adapters."""

from .pdf_loader import PDFLoader
from .text_file_document_loader import TextFileDocumentLoader
from .txt_loader import TXTLoader
from .web_loader import WebLoader

__all__ = ["TextFileDocumentLoader", "TXTLoader", "PDFLoader", "WebLoader"]
