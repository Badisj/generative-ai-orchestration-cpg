# FILE: app/main.py
"""
FastAPI application for Multimodal RAG system.
Provides endpoints for ingestion, retrieval, and query.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from app.ingestion import ingest_file, ingest_texts, save_upload
from app.retrieval import retrieve_top_k, retrieve_with_sources, Retriever
from app.generation import generate_answer, generate_answer_with_sources
from app.opensearch import get_document_stats, clear_index, delete_by_source
from app.processors import get_factory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="RAG system supporting PDFs, Word, Excel, PowerPoint, Images, and text files.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    file_type: Optional[str] = Field(default=None, description="Filter by file type (pdf, docx, etc.)")
    source_file: Optional[str] = Field(default=None, description="Filter by source filename")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[str]
    retrieved_count: int


class IngestTextRequest(BaseModel):
    """Request model for text ingestion."""
    texts: List[str] = Field(..., description="List of texts to ingest")
    source_name: str = Field(default="manual_input", description="Source identifier")


class IngestResponse(BaseModel):
    """Response model for ingestion."""
    success: bool
    message: str
    details: Dict[str, Any]


class StatsResponse(BaseModel):
    """Response model for document statistics."""
    total_documents: int
    file_types: Dict[str, int]
    source_files: List[str]


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - health check."""
    return {
        "status": "healthy",
        "service": "Multimodal RAG API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/supported-formats", tags=["Info"])
async def get_supported_formats():
    """Get list of supported file formats."""
    factory = get_factory()
    return {
        "supported_extensions": factory.supported_extensions,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
    }


# =============================================================================
# Ingestion Endpoints
# =============================================================================

@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=1000, ge=100, le=5000),
    chunk_overlap: int = Query(default=200, ge=0, le=500),
):
    """
    Upload and ingest a single file.
    
    Supported formats: PDF, DOCX, XLSX, PPTX, images, text files.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB"
        )
    
    try:
        # Save file
        file_path = save_upload(contents, file.filename)
        
        # Ingest
        result = await ingest_file(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        if result.get("success"):
            return IngestResponse(
                success=True,
                message=f"Successfully ingested {file.filename}",
                details=result,
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Ingestion failed"))
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/batch", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_batch_endpoint(
    files: List[UploadFile] = File(...),
    chunk_size: int = Query(default=1000),
    chunk_overlap: int = Query(default=200),
):
    """Upload and ingest multiple files."""
    results = []
    success_count = 0
    
    for file in files:
        try:
            contents = await file.read()
            file_path = save_upload(contents, file.filename)
            
            result = await ingest_file(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            results.append(result)
            if result.get("success"):
                success_count += 1
                
        except Exception as e:
            results.append({
                "success": False,
                "file": file.filename,
                "error": str(e),
            })
    
    return IngestResponse(
        success=success_count > 0,
        message=f"Ingested {success_count}/{len(files)} files",
        details={"files": results},
    )


@app.post("/ingest/texts", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_texts_endpoint(request: IngestTextRequest):
    """Ingest raw text strings."""
    try:
        result = await ingest_texts(
            texts=request.texts,
            source_name=request.source_name,
        )
        
        return IngestResponse(
            success=result.get("success", False),
            message=f"Ingested {result.get('chunks_stored', 0)} chunks from {len(request.texts)} texts",
            details=result,
        )
        
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Query Endpoints
# =============================================================================

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest):
    """
    Answer a question using RAG.
    
    Retrieves relevant documents and generates an answer using LLM.
    """
    # Build filters
    filters = {}
    if request.file_type:
        filters["file_type"] = request.file_type
    if request.source_file:
        filters["source_file"] = request.source_file
    
    try:
        # Retrieve documents
        results = await retrieve_with_sources(
            query=request.question,
            top_k=request.top_k,
            filters=filters if filters else None,
        )
        
        if not results:
            return QueryResponse(
                answer="No relevant documents found for your question.",
                sources=[],
                retrieved_count=0,
            )
        
        # Generate answer
        response = await generate_answer_with_sources(
            question=request.question,
            documents=results,
        )
        
        return QueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            retrieved_count=len(results),
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve", tags=["Query"])
async def retrieve_endpoint(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20),
    file_type: Optional[str] = Query(default=None),
):
    """
    Retrieve relevant documents without generation.
    
    Useful for testing retrieval quality.
    """
    filters = {"file_type": file_type} if file_type else None
    
    try:
        results = await retrieve_with_sources(
            query=query,
            top_k=top_k,
            filters=filters,
        )
        
        return {
            "query": query,
            "count": len(results),
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Management Endpoints
# =============================================================================

@app.get("/stats", response_model=StatsResponse, tags=["Management"])
async def get_stats():
    """Get document statistics."""
    try:
        stats = get_document_stats()
        return StatsResponse(
            total_documents=stats.get("total_documents", 0),
            file_types=stats.get("file_types", {}),
            source_files=stats.get("source_files", []),
        )
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{source_file}", tags=["Management"])
async def delete_source(source_file: str):
    """Delete all documents from a specific source file."""
    try:
        deleted = delete_by_source(source_file)
        return {
            "success": True,
            "deleted_count": deleted,
            "source_file": source_file,
        }
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear", tags=["Management"])
async def clear_all():
    """Clear all documents from the index. Use with caution!"""
    try:
        clear_index()
        return {"success": True, "message": "Index cleared"}
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    # Ensure upload directory exists
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Multimodal RAG API started")
