# FILE: app/retrieval/retriever.py
"""
Retriever module: embeds queries and retrieves documents from OpenSearch.
Supports filtered and hybrid search.
"""

from typing import List, Dict, Any, Optional
import logging

from app.embeddings import embed_single, embed_async
from app.opensearch import knn_query, hybrid_search

logger = logging.getLogger(__name__)


async def retrieve_top_k(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Embed query and retrieve top-k relevant documents.
    
    Parameters
    ----------
    query : str
        User query to search for.
    top_k : int
        Number of results to return.
    filters : Optional[Dict[str, Any]]
        Optional metadata filters.
        Supported filters:
        - file_type: str (e.g., "pdf", "docx")
        - file_types: List[str] (e.g., ["pdf", "xlsx"])
        - source_file: str (exact filename)
        
    Returns
    -------
    List[Dict[str, Any]]
        Retrieved documents with content, metadata, and score.
    """
    # Embed query
    query_vector = await embed_single(query)
    
    # Retrieve from OpenSearch
    results = knn_query(
        vector=query_vector.tolist(),
        k=top_k,
        filters=filters,
    )
    
    logger.info(f"Retrieved {len(results)} documents for query: '{query[:50]}...'")
    return results


async def retrieve_hybrid(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic and keyword matching.
    
    Parameters
    ----------
    query : str
        User query.
    top_k : int
        Number of results.
    filters : Optional[Dict[str, Any]]
        Optional filters.
        
    Returns
    -------
    List[Dict[str, Any]]
        Retrieved documents.
    """
    # Embed query
    query_vector = await embed_single(query)
    
    # Hybrid search
    results = hybrid_search(
        query_text=query,
        vector=query_vector.tolist(),
        k=top_k,
        filters=filters,
    )
    
    logger.info(f"Hybrid retrieved {len(results)} documents")
    return results


async def retrieve_contents(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Convenience method to retrieve just content strings.
    
    Parameters
    ----------
    query : str
        User query.
    top_k : int
        Number of results.
    filters : Optional[Dict[str, Any]]
        Optional filters.
        
    Returns
    -------
    List[str]
        List of document content strings.
    """
    results = await retrieve_top_k(query, top_k, filters)
    return [r["content"] for r in results]


async def retrieve_with_sources(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents with formatted source information.
    
    Parameters
    ----------
    query : str
        User query.
    top_k : int
        Number of results.
    filters : Optional[Dict[str, Any]]
        Optional filters.
        
    Returns
    -------
    List[Dict[str, Any]]
        Documents with content, source info, and score.
    """
    results = await retrieve_top_k(query, top_k, filters)
    
    formatted = []
    for r in results:
        metadata = r.get("metadata", {})
        source_info = metadata.get("source_file", "Unknown")
        
        if metadata.get("page_number"):
            source_info += f" (page {metadata['page_number']})"
        if metadata.get("sheet_name"):
            source_info += f" (sheet: {metadata['sheet_name']})"
        
        formatted.append({
            "content": r["content"],
            "source": source_info,
            "file_type": metadata.get("file_type", "unknown"),
            "score": r.get("score", 0),
        })
    
    return formatted


class Retriever:
    """
    Retriever class for more complex retrieval scenarios.
    """
    
    def __init__(
        self,
        top_k: int = 5,
        default_filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = False,
    ):
        """
        Initialize retriever.
        
        Parameters
        ----------
        top_k : int
            Default number of results.
        default_filters : Optional[Dict[str, Any]]
            Default filters to apply to all queries.
        use_hybrid : bool
            Whether to use hybrid search by default.
        """
        self.top_k = top_k
        self.default_filters = default_filters or {}
        self.use_hybrid = use_hybrid
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query.
        
        Parameters
        ----------
        query : str
            User query.
        top_k : Optional[int]
            Override default top_k.
        filters : Optional[Dict[str, Any]]
            Additional filters (merged with defaults).
            
        Returns
        -------
        List[Dict[str, Any]]
            Retrieved documents.
        """
        k = top_k or self.top_k
        
        # Merge filters
        merged_filters = {**self.default_filters}
        if filters:
            merged_filters.update(filters)
        
        # Use appropriate search method
        if self.use_hybrid:
            return await retrieve_hybrid(query, k, merged_filters or None)
        else:
            return await retrieve_top_k(query, k, merged_filters or None)
    
    async def retrieve_for_generation(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> tuple[List[str], List[str]]:
        """
        Retrieve documents formatted for generation.
        
        Parameters
        ----------
        query : str
            User query.
        top_k : Optional[int]
            Override default top_k.
            
        Returns
        -------
        tuple[List[str], List[str]]
            Tuple of (contents, sources).
        """
        results = await self.retrieve(query, top_k)
        
        contents = []
        sources = []
        
        for r in results:
            contents.append(r["content"])
            
            metadata = r.get("metadata", {})
            source = metadata.get("source_file", "Unknown")
            if metadata.get("page_number"):
                source += f" (p.{metadata['page_number']})"
            sources.append(source)
        
        return contents, sources
