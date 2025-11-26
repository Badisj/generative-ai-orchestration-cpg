# FILE: app/embeddings/__init__.py
"""
Embeddings module for vector generation.
"""

from app.embeddings.embedding import (
    embed_async,
    embed_sync,
    embed_single,
    get_embedder,
    get_embedding_dimension,
)

__all__ = [
    "embed_async",
    "embed_sync",
    "embed_single",
    "get_embedder",
    "get_embedding_dimension",
]
