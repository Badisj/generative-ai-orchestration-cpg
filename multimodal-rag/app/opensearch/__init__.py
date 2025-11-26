# FILE: app/opensearch/__init__.py
"""
OpenSearch client module.
"""

from app.opensearch.opensearch_client import (
    get_client,
    ensure_index_exists,
    bulk_insert,
    knn_query,
    hybrid_search,
    delete_by_source,
    get_document_stats,
    clear_index,
)

__all__ = [
    "get_client",
    "ensure_index_exists",
    "bulk_insert",
    "knn_query",
    "hybrid_search",
    "delete_by_source",
    "get_document_stats",
    "clear_index",
]
