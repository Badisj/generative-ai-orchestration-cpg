# FILE: app/ingestion/__init__.py
"""
Ingestion module for processing and storing documents.
"""

from app.ingestion.ingest import (
    ingest_file,
    ingest_texts,
    ingest_directory,
    save_upload,
)

__all__ = [
    "ingest_file",
    "ingest_texts",
    "ingest_directory",
    "save_upload",
]
