# FILE: app/chunking/chunker.py
"""
Intelligent text chunking for RAG.
Splits documents into semantically meaningful chunks with overlap.
"""

from typing import List, Optional
import logging
from dataclasses import dataclass

from app.processors.base import Document
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunking behavior."""
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    length_function: str = "characters"  # "characters" or "tokens"
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            # Ordered by preference (try to split on these first)
            self.separators = [
                "\n\n\n",     # Multiple newlines (sections)
                "\n\n",       # Double newline (paragraphs)
                "\n",         # Single newline
                ". ",         # Sentences
                "? ",
                "! ",
                "; ",
                ", ",
                " ",          # Words
                "",           # Characters (last resort)
            ]


class TextChunker:
    """
    Splits text into chunks with overlap.
    
    Features:
    - Respects semantic boundaries (paragraphs, sentences)
    - Maintains overlap for context continuity
    - Preserves metadata across chunks
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize chunker.
        
        Parameters
        ----------
        config : Optional[ChunkConfig]
            Chunking configuration.
        """
        self.config = config or ChunkConfig()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Parameters
        ----------
        text : str
            Text to chunk.
            
        Returns
        -------
        List[str]
            List of text chunks.
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # If text is smaller than chunk size, return as-is
        if len(text) <= self.config.chunk_size:
            return [text]
        
        chunks = self._recursive_split(text, self.config.separators)
        
        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)
        
        # Add overlap
        if self.config.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of documents, preserving metadata.
        
        Parameters
        ----------
        documents : List[Document]
            Documents to chunk.
            
        Returns
        -------
        List[Document]
            Chunked documents with updated metadata.
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.chunk_text(doc.content)
            
            for i, chunk in enumerate(chunks):
                # Create new document with updated metadata
                new_metadata = doc.metadata.copy()
                new_metadata["chunk_index"] = i
                new_metadata["total_chunks"] = len(chunks)
                new_metadata["original_length"] = len(doc.content)
                
                chunked_doc = Document(
                    content=chunk,
                    metadata=new_metadata,
                )
                chunked_docs.append(chunked_doc)
        
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using separators.
        
        Parameters
        ----------
        text : str
            Text to split.
        separators : List[str]
            Separators to try (in order).
            
        Returns
        -------
        List[str]
            Split text segments.
        """
        if not separators:
            # Base case: split by characters
            return self._split_by_size(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Split by characters
            return self._split_by_size(text)
        
        # Split by current separator
        parts = text.split(separator)
        
        # Process each part
        chunks = []
        for part in parts:
            if not part.strip():
                continue
            
            if len(part) <= self.config.chunk_size:
                chunks.append(part.strip())
            else:
                # Recursively split with remaining separators
                sub_chunks = self._recursive_split(part, remaining_separators)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_by_size(self, text: str) -> List[str]:
        """
        Split text by character count.
        
        Parameters
        ----------
        text : str
            Text to split.
            
        Returns
        -------
        List[str]
            Split chunks.
        """
        chunks = []
        
        while len(text) > self.config.chunk_size:
            # Find break point
            break_point = self.config.chunk_size
            
            # Try to break at a word boundary
            space_pos = text.rfind(" ", 0, break_point)
            if space_pos > break_point * 0.5:  # At least 50% through
                break_point = space_pos
            
            chunks.append(text[:break_point].strip())
            text = text[break_point:].strip()
        
        if text:
            chunks.append(text)
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that are too small.
        
        Parameters
        ----------
        chunks : List[str]
            Chunks to merge.
            
        Returns
        -------
        List[str]
            Merged chunks.
        """
        min_size = self.config.chunk_size * 0.3  # Minimum 30% of target
        merged = []
        current = ""
        
        for chunk in chunks:
            if not current:
                current = chunk
            elif len(current) + len(chunk) + 1 <= self.config.chunk_size:
                current = current + "\n\n" + chunk
            else:
                merged.append(current)
                current = chunk
        
        if current:
            # If last chunk is too small, merge with previous
            if len(current) < min_size and merged:
                if len(merged[-1]) + len(current) + 1 <= self.config.chunk_size * 1.2:
                    merged[-1] = merged[-1] + "\n\n" + current
                else:
                    merged.append(current)
            else:
                merged.append(current)
        
        return merged
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between chunks for context continuity.
        
        Parameters
        ----------
        chunks : List[str]
            Chunks to add overlap to.
            
        Returns
        -------
        List[str]
            Chunks with overlap.
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.config.chunk_overlap:]
                
                # Find word boundary in overlap
                space_pos = overlap_text.find(" ")
                if space_pos > 0:
                    overlap_text = overlap_text[space_pos + 1:]
                
                # Prepend overlap
                overlapped.append(overlap_text + " " + chunk)
        
        return overlapped


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Convenience function to chunk documents.
    
    Parameters
    ----------
    documents : List[Document]
        Documents to chunk.
    chunk_size : int
        Target chunk size in characters.
    chunk_overlap : int
        Overlap between chunks.
        
    Returns
    -------
    List[Document]
        Chunked documents.
    """
    config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunker = TextChunker(config)
    return chunker.chunk_documents(documents)
