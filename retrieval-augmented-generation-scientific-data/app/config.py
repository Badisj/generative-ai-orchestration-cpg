# FILE: app/config.py
"""
Configuration for RAG Service.
Holds OpenSearch host, index name, embedding model, and vector dimension.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenSearch configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX", "formulation_docs")

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", 768))