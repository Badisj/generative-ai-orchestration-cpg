# FILE: app/embeddings/embedding.py
"""
Async wrapper around Sentence-Transformers embedding model.
Generates embeddings for documents and queries.
"""

import asyncio
from typing import List, Union
import logging

from sentence_transformers import SentenceTransformer
import numpy as np

from app.config import EMBED_MODEL

logger = logging.getLogger(__name__)

# Load model once globally
_embedder: SentenceTransformer = None


def get_embedder() -> SentenceTransformer:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


async def embed_async(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts asynchronously.
    
    Parameters
    ----------
    texts : List[str]
        Texts to embed.
        
    Returns
    -------
    np.ndarray
        Embedding vectors as numpy array.
    """
    if not texts:
        return np.array([])
    
    embedder = get_embedder()
    loop = asyncio.get_running_loop()
    
    vectors = await loop.run_in_executor(
        None,
        lambda: embedder.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
    )
    
    return vectors


def embed_sync(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings synchronously.
    
    Parameters
    ----------
    texts : List[str]
        Texts to embed.
        
    Returns
    -------
    np.ndarray
        Embedding vectors.
    """
    if not texts:
        return np.array([])
    
    embedder = get_embedder()
    return embedder.encode(
        texts,
        convert_to_tensor=False,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


async def embed_single(text: str) -> np.ndarray:
    """
    Embed a single text.
    
    Parameters
    ----------
    text : str
        Text to embed.
        
    Returns
    -------
    np.ndarray
        Single embedding vector.
    """
    vectors = await embed_async([text])
    return vectors[0]


def get_embedding_dimension() -> int:
    """
    Get the dimension of embeddings.
    
    Returns
    -------
    int
        Embedding dimension.
    """
    embedder = get_embedder()
    return embedder.get_sentence_embedding_dimension()
