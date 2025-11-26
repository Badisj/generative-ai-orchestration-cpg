# FILE: app/processors/__init__.py
"""
Document processors package.
Handles extraction of text from various file formats.
"""

from app.processors.base import BaseProcessor, Document
from app.processors.factory import (
    ProcessorFactory,
    get_factory,
    process_file,
)
from app.processors.pdf_processor import PDFProcessor
from app.processors.docx_processor import DocxProcessor
from app.processors.xlsx_processor import ExcelProcessor
from app.processors.pptx_processor import PowerPointProcessor
from app.processors.image_processor import ImageProcessor
from app.processors.text_processor import TextProcessor

__all__ = [
    "BaseProcessor",
    "Document",
    "ProcessorFactory",
    "get_factory",
    "process_file",
    "PDFProcessor",
    "DocxProcessor",
    "ExcelProcessor",
    "PowerPointProcessor",
    "ImageProcessor",
    "TextProcessor",
]
