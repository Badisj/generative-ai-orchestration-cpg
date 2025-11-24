# FILE: app/embeddings/embedding.py
"""
Async wrapper around Sentence-Transformers embedding model.
Generates embeddings for a list of texts asynchronously.
"""

import asyncio
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL

# Load model once globally
_embedder = SentenceTransformer(EMBED_MODEL)

# async def embed_async(texts):
#     """Return embeddings for list of texts asynchronously."""
#     loop = asyncio.get_running_loop()
#     vectors = await loop.run_in_executor(None, _embedder.encode, texts, convert_to_tensor=True, show_progress_bar=True)
#     return vectors


async def embed_async(texts: list[str]):
    """Return embeddings for list of texts asynchronously."""
    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(
        None,
        lambda: _embedder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
    )
    return vectors