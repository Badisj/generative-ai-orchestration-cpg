# FILE: app/processors/pdf_processor.py
"""
PDF processor using pymupdf4llm for structured markdown extraction.
Handles both text-based and scanned PDFs.
"""

from pathlib import Path
from typing import List, Optional
import logging

from app.processors.base import BaseProcessor, Document
from app.config import OCR_ENABLED

logger = logging.getLogger(__name__)


class PDFProcessor(BaseProcessor):
    """
    Process PDF files using pymupdf4llm.
    
    Features:
    - Extracts text with markdown formatting (headers, tables, lists)
    - Preserves document structure
    - Handles multi-page documents
    - OCR fallback for scanned pages (optional)
    """
    
    supported_extensions = [".pdf"]
    
    def __init__(self, ocr_enabled: bool = OCR_ENABLED):
        """
        Initialize PDF processor.
        
        Parameters
        ----------
        ocr_enabled : bool
            Whether to use OCR for scanned pages.
        """
        self.ocr_enabled = ocr_enabled
    
    def can_process(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def process(self, file_path: Path) -> List[Document]:
        """
        Extract text from PDF with structure preservation.
        
        Parameters
        ----------
        file_path : Path
            Path to PDF file.
            
        Returns
        -------
        List[Document]
            List of documents, one per page with extracted content.
        """
        import pymupdf4llm
        import pymupdf
        
        file_path = Path(file_path)
        documents = []
        
        try:
            # Extract markdown with page information
            md_text = pymupdf4llm.to_markdown(
                str(file_path),
                page_chunks=True,  # Get per-page chunks
            )
            
            # md_text is a list of dicts when page_chunks=True
            if isinstance(md_text, list):
                for page_data in md_text:
                    content = page_data.get("text", "").strip()
                    page_num = page_data.get("metadata", {}).get("page", 1)
                    
                    if content:
                        doc = self._create_document(
                            content=content,
                            source_file=file_path.name,
                            file_type="pdf",
                            page_number=page_num,
                            extraction_method="pymupdf4llm",
                        )
                        documents.append(doc)
            else:
                # Single string result
                if md_text.strip():
                    doc = self._create_document(
                        content=md_text.strip(),
                        source_file=file_path.name,
                        file_type="pdf",
                        page_number=1,
                        extraction_method="pymupdf4llm",
                    )
                    documents.append(doc)
            
            # If no content extracted, try basic PyMuPDF
            if not documents:
                documents = self._fallback_extraction(file_path)
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            # Try fallback extraction
            documents = self._fallback_extraction(file_path)
        
        logger.info(f"Extracted {len(documents)} pages from {file_path.name}")
        return documents
    
    def _fallback_extraction(self, file_path: Path) -> List[Document]:
        """
        Fallback extraction using basic PyMuPDF.
        
        Parameters
        ----------
        file_path : Path
            Path to PDF file.
            
        Returns
        -------
        List[Document]
            Extracted documents.
        """
        import pymupdf
        
        documents = []
        
        try:
            pdf = pymupdf.open(str(file_path))
            
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                
                # If text is empty and OCR is enabled, try OCR
                if not text and self.ocr_enabled:
                    text = self._ocr_page(page)
                
                if text:
                    doc = self._create_document(
                        content=text,
                        source_file=file_path.name,
                        file_type="pdf",
                        page_number=page_num,
                        extraction_method="pymupdf_fallback",
                    )
                    documents.append(doc)
            
            pdf.close()
            
        except Exception as e:
            logger.error(f"Fallback extraction failed for {file_path}: {e}")
        
        return documents
    
    def _ocr_page(self, page) -> str:
        """
        OCR a single page using pytesseract.
        
        Parameters
        ----------
        page : pymupdf.Page
            PyMuPDF page object.
            
        Returns
        -------
        str
            OCR extracted text.
        """
        try:
            import pytesseract
            from PIL import Image
            import io
            
            # Render page to image
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # OCR
            text = pytesseract.image_to_string(img)
            return text.strip()
            
        except ImportError:
            logger.warning("pytesseract not installed, skipping OCR")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""
