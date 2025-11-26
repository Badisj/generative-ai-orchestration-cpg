# FILE: app/processors/factory.py
"""
Processor factory for automatic file type detection and routing.
"""

from pathlib import Path
from typing import List, Optional, Type
import logging

from app.processors.base import BaseProcessor, Document
from app.processors.pdf_processor import PDFProcessor
from app.processors.docx_processor import DocxProcessor
from app.processors.xlsx_processor import ExcelProcessor
from app.processors.pptx_processor import PowerPointProcessor
from app.processors.image_processor import ImageProcessor
from app.processors.text_processor import TextProcessor
from app.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """
    Factory for creating and managing document processors.
    
    Automatically detects file type and routes to appropriate processor.
    """
    
    def __init__(self):
        """Initialize factory with all available processors."""
        self._processors: List[BaseProcessor] = [
            PDFProcessor(),
            DocxProcessor(),
            ExcelProcessor(),
            PowerPointProcessor(),
            ImageProcessor(),
            TextProcessor(),
        ]
        
        # Build extension to processor mapping
        self._extension_map: dict[str, BaseProcessor] = {}
        for processor in self._processors:
            for ext in processor.supported_extensions:
                self._extension_map[ext.lower()] = processor
    
    def get_processor(self, file_path: Path) -> Optional[BaseProcessor]:
        """
        Get appropriate processor for a file.
        
        Parameters
        ----------
        file_path : Path
            Path to file.
            
        Returns
        -------
        Optional[BaseProcessor]
            Processor instance or None if unsupported.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        return self._extension_map.get(ext)
    
    def process_file(self, file_path: Path) -> List[Document]:
        """
        Process a file using the appropriate processor.
        
        Parameters
        ----------
        file_path : Path
            Path to file.
            
        Returns
        -------
        List[Document]
            Processed documents.
            
        Raises
        ------
        ValueError
            If file type is not supported.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        processor = self.get_processor(file_path)
        
        if processor is None:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        
        logger.info(f"Processing {file_path.name} with {processor.__class__.__name__}")
        
        return processor.process(file_path)
    
    def is_supported(self, file_path: Path) -> bool:
        """
        Check if a file type is supported.
        
        Parameters
        ----------
        file_path : Path
            Path to file.
            
        Returns
        -------
        bool
            True if file type is supported.
        """
        ext = Path(file_path).suffix.lower()
        return ext in self._extension_map
    
    @property
    def supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        return list(self._extension_map.keys())
    
    def register_processor(self, processor: BaseProcessor):
        """
        Register a custom processor.
        
        Parameters
        ----------
        processor : BaseProcessor
            Processor instance to register.
        """
        self._processors.append(processor)
        for ext in processor.supported_extensions:
            self._extension_map[ext.lower()] = processor
        logger.info(f"Registered {processor.__class__.__name__} for {processor.supported_extensions}")


# Global factory instance
_factory: Optional[ProcessorFactory] = None


def get_factory() -> ProcessorFactory:
    """Get or create the global processor factory."""
    global _factory
    if _factory is None:
        _factory = ProcessorFactory()
    return _factory


def process_file(file_path: Path) -> List[Document]:
    """
    Convenience function to process a file.
    
    Parameters
    ----------
    file_path : Path
        Path to file to process.
        
    Returns
    -------
    List[Document]
        Processed documents.
    """
    return get_factory().process_file(file_path)
