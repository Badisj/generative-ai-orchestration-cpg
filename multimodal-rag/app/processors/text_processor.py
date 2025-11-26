# FILE: app/processors/text_processor.py
"""
Plain text and Markdown processor.
Simple extraction with minimal processing.
"""

from pathlib import Path
from typing import List
import logging

from app.processors.base import BaseProcessor, Document

logger = logging.getLogger(__name__)


class TextProcessor(BaseProcessor):
    """
    Process plain text and Markdown files.
    
    Features:
    - Handles .txt, .md, .markdown files
    - Preserves original formatting
    - Handles various encodings
    """
    
    supported_extensions = [".txt", ".md", ".markdown", ".rst", ".text"]
    
    def __init__(self, encodings: List[str] = None):
        """
        Initialize text processor.
        
        Parameters
        ----------
        encodings : List[str]
            List of encodings to try (in order).
        """
        self.encodings = encodings or ["utf-8", "utf-16", "latin-1", "cp1252"]
    
    def can_process(self, file_path: Path) -> bool:
        """Check if file is a text file."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def process(self, file_path: Path) -> List[Document]:
        """
        Read and process text file.
        
        Parameters
        ----------
        file_path : Path
            Path to text file.
            
        Returns
        -------
        List[Document]
            Single document with file contents.
        """
        file_path = Path(file_path)
        content = None
        used_encoding = None
        
        # Try different encodings
        for encoding in self.encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path} with {encoding}: {e}")
                continue
        
        if content is None:
            logger.error(f"Could not read {file_path} with any encoding")
            return []
        
        content = content.strip()
        
        if content:
            # Determine file type
            suffix = file_path.suffix.lower()
            if suffix in [".md", ".markdown"]:
                file_type = "markdown"
            elif suffix == ".rst":
                file_type = "rst"
            else:
                file_type = "text"
            
            doc = self._create_document(
                content=content,
                source_file=file_path.name,
                file_type=file_type,
                encoding=used_encoding,
                char_count=len(content),
                line_count=content.count("\n") + 1,
            )
            logger.info(f"Processed text file {file_path.name}")
            return [doc]
        
        return []
