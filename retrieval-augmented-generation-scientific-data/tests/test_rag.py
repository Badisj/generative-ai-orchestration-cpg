# FILE: tests/test_rag.py
"""
Basic async smoke tests for the RAG service.
"""

# Ensure repo root is on sys.path when running this test as a script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
from app.retrieval.retriever import retrieve_top_k
from app.generation.generator import generate_answer


query_text = "What are the formulations discussed here?"
top_k = 10

async def query_rag(q, k):
    # Retrieve relevant documents
    retrieved_docs = await retrieve_top_k(q, k)
    print("\nTop-k retrieved documents:")
    for doc in retrieved_docs:
        print("-", doc)
    
    # Generate answer from retrieved docs
    answer = await generate_answer(q, retrieved_docs)
    print("\nGenerated answer:")
    print(answer)

# await query_rag(query_text, top_k)

# Run test standalone
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="What are the formulations discussed here?")
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    asyncio.run(query_rag(args.query, args.k))