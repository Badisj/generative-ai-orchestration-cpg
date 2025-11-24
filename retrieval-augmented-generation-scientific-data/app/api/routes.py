# FILE: app/api/routes.py
"""
FastAPI API routes for RAG backend.
Provides endpoints for document ingestion and question answering.
"""

from fastapi import APIRouter, Query
from app.ingestion.ingest import ingest_texts
from app.retrieval.retriever import retrieve_top_k
from app.generation.generator import generate_answer

router = APIRouter(prefix="/rag")

@router.post("/insert")
async def insert_docs(docs: list[str]):
    """Insert a list of text documents asynchronously."""
    count = await ingest_texts(docs)
    return {"inserted": count}

@router.get("/query")
async def query(q: str = Query(..., description="Query string"), k: int = Query(5, description="Top k results")):
    """Retrieve context and generate answer for a query."""
    docs = await retrieve_top_k(q, k)
    answer = await generate_answer(q, docs)
    return {"answer": answer, "context": docs}
