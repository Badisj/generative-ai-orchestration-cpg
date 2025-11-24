# FILE: app/main.py
"""
Main FastAPI entry point for the RAG service.
Mounts API routes and health check endpoint.
"""

from fastapi import FastAPI
from app.api.routes import router

# Create FastAPI app
app = FastAPI(title="RAG Service (OpenSearch)")

# Include API routes
app.include_router(router)

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}