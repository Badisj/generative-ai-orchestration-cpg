# FILE: app/opensearch/opensearch_client.py
"""
OpenSearch client wrapper with enhanced schema for multimodal documents.
Provides bulk insertion, KNN vector search, and filtered queries.
"""

from typing import List, Dict, Any, Optional
import logging

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError

from app.config import OPENSEARCH_HOST, INDEX_NAME, EMBED_DIM

logger = logging.getLogger(__name__)

# Initialize OpenSearch client
client: OpenSearch = None


def get_client() -> OpenSearch:
    """Get or create OpenSearch client."""
    global client
    if client is None:
        client = OpenSearch(
            hosts=[OPENSEARCH_HOST],
            http_compress=True,
            timeout=60,
        )
        logger.info(f"Connected to OpenSearch at {OPENSEARCH_HOST}")
    return client


def ensure_index_exists(index_name: str = INDEX_NAME):
    """
    Ensure the vector index exists with proper schema.
    
    Parameters
    ----------
    index_name : str
        Name of the index.
    """
    client = get_client()
    
    if not client.indices.exists(index=index_name):
        logger.info(f"Creating index: {index_name}")
        
        client.indices.create(
            index=index_name,
            body={
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100,
                    },
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard",
                        },
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": EMBED_DIM,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24,
                                },
                            },
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "source_file": {"type": "keyword"},
                                "file_type": {"type": "keyword"},
                                "page_number": {"type": "integer"},
                                "chunk_index": {"type": "integer"},
                                "total_chunks": {"type": "integer"},
                                "timestamp": {"type": "date"},
                                "sheet_name": {"type": "keyword"},
                                "extraction_method": {"type": "keyword"},
                            },
                        },
                    },
                },
            },
        )
        logger.info(f"Index {index_name} created successfully")
    else:
        logger.info(f"Index {index_name} already exists")


def bulk_insert(
    documents: List[Dict[str, Any]],
    index_name: str = INDEX_NAME
) -> int:
    """
    Insert multiple documents with embeddings into OpenSearch.
    
    Parameters
    ----------
    documents : List[Dict[str, Any]]
        List of document dicts with content, embedding, and metadata.
    index_name : str
        Target index name.
        
    Returns
    -------
    int
        Number of documents inserted.
    """
    if not documents:
        return 0
    
    client = get_client()
    ensure_index_exists(index_name)
    
    ops = []
    for doc in documents:
        ops.append({"index": {"_index": index_name}})
        ops.append({
            "content": doc.get("content", ""),
            "embedding": doc.get("embedding", []),
            "metadata": doc.get("metadata", {}),
        })
    
    response = client.bulk(body=ops, refresh=True)
    
    # Count successful insertions
    success_count = 0
    if "items" in response:
        for item in response["items"]:
            if item.get("index", {}).get("status") in [200, 201]:
                success_count += 1
    
    logger.info(f"Inserted {success_count}/{len(documents)} documents")
    return success_count


def knn_query(
    vector: List[float],
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    index_name: str = INDEX_NAME
) -> List[Dict[str, Any]]:
    """
    Perform KNN search for top-k similar documents.
    
    Parameters
    ----------
    vector : List[float]
        Query embedding vector.
    k : int
        Number of results to return.
    filters : Optional[Dict[str, Any]]
        Optional metadata filters (e.g., {"file_type": "pdf"}).
    index_name : str
        Index to search.
        
    Returns
    -------
    List[Dict[str, Any]]
        List of matching documents with content, metadata, and score.
    """
    client = get_client()
    
    # Build query
    if filters:
        # KNN with filter
        query = {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": vector,
                                    "k": k,
                                }
                            }
                        }
                    ],
                    "filter": _build_filters(filters),
                }
            },
        }
    else:
        # Simple KNN
        query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector,
                        "k": k,
                    }
                }
            },
        }
    
    try:
        response = client.search(index=index_name, body=query)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "score": hit["_score"],
                "id": hit["_id"],
            })
        
        return results
        
    except Exception as e:
        logger.error(f"KNN query failed: {e}")
        return []


def _build_filters(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build OpenSearch filter clauses from filter dict.
    
    Parameters
    ----------
    filters : Dict[str, Any]
        Filter conditions.
        
    Returns
    -------
    List[Dict[str, Any]]
        OpenSearch filter clauses.
    """
    filter_clauses = []
    
    for key, value in filters.items():
        if key in ["source_file", "file_type", "extraction_method"]:
            filter_clauses.append({
                "term": {f"metadata.{key}": value}
            })
        elif key == "file_types" and isinstance(value, list):
            filter_clauses.append({
                "terms": {"metadata.file_type": value}
            })
    
    return filter_clauses


def hybrid_search(
    query_text: str,
    vector: List[float],
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    index_name: str = INDEX_NAME
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining KNN and text search.
    
    Parameters
    ----------
    query_text : str
        Text query for BM25 matching.
    vector : List[float]
        Query embedding for KNN.
    k : int
        Number of results.
    filters : Optional[Dict[str, Any]]
        Optional filters.
    index_name : str
        Index to search.
        
    Returns
    -------
    List[Dict[str, Any]]
        Search results.
    """
    client = get_client()
    
    # Hybrid query combining KNN and text match
    query = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": vector,
                                "k": k,
                            }
                        }
                    },
                    {
                        "match": {
                            "content": {
                                "query": query_text,
                                "boost": 0.3,  # Lower weight for text match
                            }
                        }
                    }
                ],
                "filter": _build_filters(filters) if filters else [],
            }
        },
    }
    
    try:
        response = client.search(index=index_name, body=query)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "score": hit["_score"],
                "id": hit["_id"],
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []


def delete_by_source(source_file: str, index_name: str = INDEX_NAME) -> int:
    """
    Delete all documents from a specific source file.
    
    Parameters
    ----------
    source_file : str
        Source filename to delete.
    index_name : str
        Index name.
        
    Returns
    -------
    int
        Number of deleted documents.
    """
    client = get_client()
    
    try:
        response = client.delete_by_query(
            index=index_name,
            body={
                "query": {
                    "term": {
                        "metadata.source_file": source_file
                    }
                }
            },
            refresh=True,
        )
        deleted = response.get("deleted", 0)
        logger.info(f"Deleted {deleted} documents from source: {source_file}")
        return deleted
        
    except Exception as e:
        logger.error(f"Delete by source failed: {e}")
        return 0


def get_document_stats(index_name: str = INDEX_NAME) -> Dict[str, Any]:
    """
    Get statistics about indexed documents.
    
    Parameters
    ----------
    index_name : str
        Index name.
        
    Returns
    -------
    Dict[str, Any]
        Statistics including document count, file types, etc.
    """
    client = get_client()
    
    try:
        # Get document count
        count_response = client.count(index=index_name)
        total_docs = count_response.get("count", 0)
        
        # Get file type distribution
        agg_response = client.search(
            index=index_name,
            body={
                "size": 0,
                "aggs": {
                    "file_types": {
                        "terms": {
                            "field": "metadata.file_type",
                            "size": 20,
                        }
                    },
                    "source_files": {
                        "terms": {
                            "field": "metadata.source_file",
                            "size": 100,
                        }
                    },
                },
            },
        )
        
        file_types = {
            bucket["key"]: bucket["doc_count"]
            for bucket in agg_response.get("aggregations", {}).get("file_types", {}).get("buckets", [])
        }
        
        source_files = [
            bucket["key"]
            for bucket in agg_response.get("aggregations", {}).get("source_files", {}).get("buckets", [])
        ]
        
        return {
            "total_documents": total_docs,
            "file_types": file_types,
            "source_files": source_files,
            "index_name": index_name,
        }
        
    except NotFoundError:
        return {
            "total_documents": 0,
            "file_types": {},
            "source_files": [],
            "index_name": index_name,
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"error": str(e)}


def clear_index(index_name: str = INDEX_NAME):
    """
    Delete and recreate the index.
    
    Parameters
    ----------
    index_name : str
        Index to clear.
    """
    client = get_client()
    
    try:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
            logger.info(f"Deleted index: {index_name}")
        
        ensure_index_exists(index_name)
        
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        raise


# Initialize on module load
ensure_index_exists()
