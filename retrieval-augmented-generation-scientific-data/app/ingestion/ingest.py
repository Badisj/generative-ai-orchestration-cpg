# FILE: app/ingestion/ingest.py
"""
Ingestion module.
Async embedding and insertion of raw text documents into OpenSearch.
"""

from app.embeddings.embedding import embed_async
from app.opensearch.opensearch_client import bulk_insert

async def ingest_texts(texts: list[str]) -> int:
    """Embed and insert a list of texts into OpenSearch.

    Returns the number of documents ingested.
    """
    vectors = await embed_async(texts)
    bulk_insert(texts, vectors)
    return len(texts)
