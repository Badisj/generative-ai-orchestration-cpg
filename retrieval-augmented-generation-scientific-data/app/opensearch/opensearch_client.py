# FILE: app/opensearch/opensearch_client.py
"""
OpenSearch client wrapper.
Provides bulk insertion and KNN vector search.
Ensures index exists on startup.
"""

from opensearchpy import OpenSearch
from app.config import OPENSEARCH_HOST, INDEX_NAME, EMBED_DIM

# Initialize OpenSearch client
client = OpenSearch(hosts=[OPENSEARCH_HOST], http_compress=True, timeout=60)

# Ensure vector index exists
if not client.indices.exists(index=INDEX_NAME):
    client.indices.create(
        index=INDEX_NAME,
        body={
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {"type": "knn_vector", "dimension": EMBED_DIM},
                }
            }
        }
    )

def bulk_insert(texts, vectors):
    """Insert multiple documents with embeddings into OpenSearch."""
    ops = []
    for t, v in zip(texts, vectors):
        ops.append({"index": {"_index": INDEX_NAME}})
        ops.append({"content": t, "embedding": v.tolist()})
    client.bulk(body=ops, refresh=True)

def knn_query(vector, k):
    """Perform KNN search for top-k similar documents."""
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": vector.tolist(), "k": k}}},
        }
    )
    return [hit["_source"]["content"] for hit in response["hits"]["hits"]]
