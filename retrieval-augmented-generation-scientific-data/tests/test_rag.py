# FILE: tests/test_rag.py
"""
Basic async smoke tests for the RAG service.
"""

import asyncio
from app.retrieval.retriever import retrieve_top_k

async def test_retrieve():
    """Test top-k retrieval returns a list and respects k."""
    q = "example formulation question"
    top_k = 2
    results = await retrieve_top_k(q, top_k)
    assert isinstance(results, list)
    assert len(results) <= top_k

# Run test standalone
if __name__ == '__main__':
    asyncio.run(test_retrieve())
