# FILE: app/processors/base.py
"""
Base processor interface and Document schema for multimodal RAG.
All document processors inherit from BaseProcessor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path


@dataclass
class Document:
    """
    Represents a processed document chunk with metadata.
    
    Attributes
    ----------
    content : str
        The extracted text content of this chunk.
    metadata : Dict[str, Any]
        Metadata about the document and chunk.
    embedding : Optional[List[float]]
        Vector embedding (populated during ingestion).
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Ensure metadata has required fields."""
        defaults = {
            "source_file": "",
            "file_type": "",
            "chunk_index": 0,
            "total_chunks": 1,
            "page_number": None,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for key, value in defaults.items():
            self.metadata.setdefault(key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for OpenSearch insertion."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        return cls(
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


class BaseProcessor(ABC):
    """
    Abstract base class for document processors.
    
    Each processor handles a specific file type and extracts
    text content with relevant metadata.
    """
    
    # File extensions this processor handles
    supported_extensions: List[str] = []
    
    @abstractmethod
    def process(self, file_path: Path) -> List[Document]:
        """
        Process a file and return list of Document objects.
        
        Parameters
        ----------
        file_path : Path
            Path to the file to process.
            
        Returns
        -------
        List[Document]
            List of documents with extracted content and metadata.
        """
        pass
    
    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """
        Check if this processor can handle the given file.
        
        Parameters
        ----------
        file_path : Path
            Path to the file to check.
            
        Returns
        -------
        bool
            True if this processor can handle the file.
        """
        pass
    
    def _create_document(
        self,
        content: str,
        source_file: str,
        file_type: str,
        page_number: Optional[int] = None,
        **extra_metadata
    ) -> Document:
        """
        Helper to create a Document with standard metadata.
        
        Parameters
        ----------
        content : str
            Extracted text content.
        source_file : str
            Original filename.
        file_type : str
            Type of file (pdf, docx, etc.).
        page_number : Optional[int]
            Page number if applicable.
        **extra_metadata
            Additional metadata fields.
            
        Returns
        -------
        Document
            Document object with metadata.
        """
        metadata = {
            "source_file": source_file,
            "file_type": file_type,
            "page_number": page_number,
            "timestamp": datetime.utcnow().isoformat(),
            **extra_metadata,
        }
        return Document(content=content, metadata=metadata)
