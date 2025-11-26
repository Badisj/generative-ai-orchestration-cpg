# FILE: tests/test_processors.py
"""
Tests for document processors.
"""

import pytest
from pathlib import Path
from app.processors import (
    get_factory,
    PDFProcessor,
    DocxProcessor,
    ExcelProcessor,
    TextProcessor,
    ImageProcessor,
)
from app.processors.base import Document


class TestProcessorFactory:
    """Tests for ProcessorFactory."""
    
    def test_factory_initialization(self):
        """Test factory creates with all processors."""
        factory = get_factory()
        assert factory is not None
        assert len(factory.supported_extensions) > 0
    
    def test_pdf_processor_detection(self):
        """Test PDF processor detection."""
        factory = get_factory()
        processor = factory.get_processor(Path("test.pdf"))
        assert processor is not None
        assert isinstance(processor, PDFProcessor)
    
    def test_docx_processor_detection(self):
        """Test DOCX processor detection."""
        factory = get_factory()
        processor = factory.get_processor(Path("test.docx"))
        assert processor is not None
        assert isinstance(processor, DocxProcessor)
    
    def test_xlsx_processor_detection(self):
        """Test Excel processor detection."""
        factory = get_factory()
        processor = factory.get_processor(Path("test.xlsx"))
        assert processor is not None
        assert isinstance(processor, ExcelProcessor)
    
    def test_image_processor_detection(self):
        """Test image processor detection."""
        factory = get_factory()
        for ext in [".png", ".jpg", ".jpeg", ".gif"]:
            processor = factory.get_processor(Path(f"test{ext}"))
            assert processor is not None
            assert isinstance(processor, ImageProcessor)
    
    def test_unsupported_format(self):
        """Test unsupported format returns None."""
        factory = get_factory()
        processor = factory.get_processor(Path("test.xyz"))
        assert processor is None
    
    def test_is_supported(self):
        """Test is_supported method."""
        factory = get_factory()
        assert factory.is_supported(Path("test.pdf")) is True
        assert factory.is_supported(Path("test.docx")) is True
        assert factory.is_supported(Path("test.xyz")) is False


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(content="Test content")
        assert doc.content == "Test content"
        assert "source_file" in doc.metadata
    
    def test_document_with_metadata(self):
        """Test document with custom metadata."""
        doc = Document(
            content="Test content",
            metadata={
                "source_file": "test.pdf",
                "file_type": "pdf",
                "page_number": 1,
            }
        )
        assert doc.metadata["source_file"] == "test.pdf"
        assert doc.metadata["file_type"] == "pdf"
    
    def test_document_to_dict(self):
        """Test document conversion to dict."""
        doc = Document(
            content="Test content",
            metadata={"source_file": "test.pdf"},
        )
        d = doc.to_dict()
        assert "content" in d
        assert "metadata" in d
        assert d["content"] == "Test content"


class TestTextProcessor:
    """Tests for TextProcessor."""
    
    def test_can_process_txt(self):
        """Test text processor recognizes .txt files."""
        processor = TextProcessor()
        assert processor.can_process(Path("test.txt")) is True
        assert processor.can_process(Path("test.md")) is True
        assert processor.can_process(Path("test.pdf")) is False
