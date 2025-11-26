# FILE: app/config.py
"""
Configuration settings for Multimodal RAG application.
Loads from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# OpenSearch Configuration
# =============================================================================
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "multimodal_docs")

# =============================================================================
# Embedding Configuration
# =============================================================================
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", 384))

# =============================================================================
# LLM Configuration
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")

# =============================================================================
# Chunking Configuration
# =============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# =============================================================================
# File Upload Configuration
# =============================================================================
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx", ".doc",
    ".xlsx", ".xls", ".csv",
    ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".txt", ".md",
}

# =============================================================================
# Processing Configuration
# =============================================================================
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
VISION_DESCRIPTION_ENABLED = os.getenv("VISION_DESCRIPTION_ENABLED", "true").lower() == "true"
