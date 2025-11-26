# FILE: app/chunking/__init__.py
"""
Chunking module for splitting documents into manageable pieces.
"""

from app.chunking.chunker import (
    TextChunker,
    ChunkConfig,
    chunk_documents,
)

__all__ = [
    "TextChunker",
    "ChunkConfig",
    "chunk_documents",
]
