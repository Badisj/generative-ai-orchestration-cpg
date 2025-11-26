# FILE: app/ingestion/ingest.py
"""
Unified ingestion pipeline for multimodal documents.
Handles: file upload → process → chunk → embed → store
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging
import shutil
from datetime import datetime

from app.processors import process_file, Document, get_factory
from app.chunking import chunk_documents
from app.embeddings import embed_async
from app.opensearch import bulk_insert, delete_by_source
from app.config import UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


async def ingest_file(
    file_path: Union[str, Path],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    replace_existing: bool = True,
) -> Dict[str, Any]:
    """
    Ingest a single file into the RAG system.
    
    Pipeline:
    1. Process file to extract text (using appropriate processor)
    2. Chunk the extracted text
    3. Generate embeddings for each chunk
    4. Store in OpenSearch with metadata
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to file to ingest.
    chunk_size : int
        Target chunk size in characters.
    chunk_overlap : int
        Overlap between chunks.
    replace_existing : bool
        If True, delete existing documents from same source.
        
    Returns
    -------
    Dict[str, Any]
        Ingestion results with document count and metadata.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Ingesting file: {file_path.name}")
    start_time = datetime.utcnow()
    
    # Step 1: Process file
    try:
        documents = process_file(file_path)
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "file": file_path.name,
        }
    
    if not documents:
        return {
            "success": False,
            "error": "No content extracted from file",
            "file": file_path.name,
        }
    
    logger.info(f"Extracted {len(documents)} documents from {file_path.name}")
    
    # Step 2: Chunk documents
    chunked_docs = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    logger.info(f"Created {len(chunked_docs)} chunks")
    
    # Step 3: Generate embeddings
    contents = [doc.content for doc in chunked_docs]
    vectors = await embed_async(contents)
    
    # Attach embeddings to documents
    docs_with_embeddings = []
    for doc, vec in zip(chunked_docs, vectors):
        doc_dict = doc.to_dict()
        doc_dict["embedding"] = vec.tolist()
        docs_with_embeddings.append(doc_dict)
    
    # Step 4: Delete existing if requested
    if replace_existing:
        deleted = delete_by_source(file_path.name)
        if deleted > 0:
            logger.info(f"Deleted {deleted} existing documents")
    
    # Step 5: Store in OpenSearch
    inserted = bulk_insert(docs_with_embeddings)
    
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    
    return {
        "success": True,
        "file": file_path.name,
        "file_type": documents[0].metadata.get("file_type", "unknown"),
        "documents_extracted": len(documents),
        "chunks_created": len(chunked_docs),
        "chunks_stored": inserted,
        "processing_time_seconds": round(elapsed, 2),
    }


async def ingest_texts(
    texts: List[str],
    source_name: str = "manual_input",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ingest raw text strings into the RAG system.
    
    Parameters
    ----------
    texts : List[str]
        List of text strings to ingest.
    source_name : str
        Source identifier for these texts.
    metadata : Optional[Dict[str, Any]]
        Additional metadata to attach.
        
    Returns
    -------
    Dict[str, Any]
        Ingestion results.
    """
    if not texts:
        return {"success": False, "error": "No texts provided"}
    
    logger.info(f"Ingesting {len(texts)} text strings")
    
    # Create documents
    documents = []
    base_metadata = metadata or {}
    
    for i, text in enumerate(texts):
        if text.strip():
            doc = Document(
                content=text.strip(),
                metadata={
                    "source_file": source_name,
                    "file_type": "text",
                    "chunk_index": i,
                    "total_chunks": len(texts),
                    **base_metadata,
                }
            )
            documents.append(doc)
    
    # Chunk if texts are long
    chunked_docs = chunk_documents(documents)
    
    # Generate embeddings
    contents = [doc.content for doc in chunked_docs]
    vectors = await embed_async(contents)
    
    # Prepare for storage
    docs_with_embeddings = []
    for doc, vec in zip(chunked_docs, vectors):
        doc_dict = doc.to_dict()
        doc_dict["embedding"] = vec.tolist()
        docs_with_embeddings.append(doc_dict)
    
    # Store
    inserted = bulk_insert(docs_with_embeddings)
    
    return {
        "success": True,
        "source": source_name,
        "texts_provided": len(texts),
        "chunks_stored": inserted,
    }


async def ingest_directory(
    directory_path: Union[str, Path],
    recursive: bool = True,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """
    Ingest all supported files from a directory.
    
    Parameters
    ----------
    directory_path : Union[str, Path]
        Path to directory.
    recursive : bool
        Whether to process subdirectories.
    chunk_size : int
        Target chunk size.
    chunk_overlap : int
        Chunk overlap.
        
    Returns
    -------
    Dict[str, Any]
        Batch ingestion results.
    """
    directory_path = Path(directory_path)
    
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")
    
    factory = get_factory()
    results = {
        "total_files": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "files": [],
    }
    
    # Collect files
    if recursive:
        files = list(directory_path.rglob("*"))
    else:
        files = list(directory_path.glob("*"))
    
    # Filter to supported files
    supported_files = [f for f in files if f.is_file() and factory.is_supported(f)]
    results["total_files"] = len(supported_files)
    
    logger.info(f"Found {len(supported_files)} supported files in {directory_path}")
    
    # Process each file
    for file_path in supported_files:
        try:
            file_result = await ingest_file(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            if file_result.get("success"):
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            results["files"].append(file_result)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            results["failed"] += 1
            results["files"].append({
                "success": False,
                "file": file_path.name,
                "error": str(e),
            })
    
    return results


def save_upload(
    file_content: bytes,
    filename: str,
    upload_dir: str = UPLOAD_DIR,
) -> Path:
    """
    Save uploaded file content to disk.
    
    Parameters
    ----------
    file_content : bytes
        File content as bytes.
    filename : str
        Original filename.
    upload_dir : str
        Directory to save to.
        
    Returns
    -------
    Path
        Path to saved file.
    """
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename if exists
    file_path = upload_path / filename
    if file_path.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        counter = 1
        while file_path.exists():
            file_path = upload_path / f"{stem}_{counter}{suffix}"
            counter += 1
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    logger.info(f"Saved upload: {file_path}")
    return file_path
