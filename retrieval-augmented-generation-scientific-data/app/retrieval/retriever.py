# FILE: app/retrieval/retriever.py
"""
Retriever module: embeds queries and retrieves top-k documents from OpenSearch.
"""

from app.embeddings.embedding import embed_async
from app.opensearch.opensearch_client import knn_query

async def retrieve_top_k(query: str, top_k: int):
    """Embed query and retrieve top-k relevant documents."""
    # Embed query asynchronously
    vec = (await embed_async([query]))[0]
    # Retrieve top-k documents using OpenSearch KNN
    results = knn_query(vec, top_k)
    return results
