# Multimodal RAG System

A production-ready Retrieval-Augmented Generation (RAG) system supporting multiple document formats including PDFs, Word documents, Excel spreadsheets, PowerPoint presentations, images, and text files.

## Features

- **Multi-format Document Processing**: PDF, DOCX, XLSX, PPTX, PNG/JPG, TXT/MD
- **Intelligent Chunking**: Semantic text splitting with configurable overlap
- **Vector Search**: OpenSearch with HNSW indexing for fast KNN queries
- **Hybrid Search**: Combines semantic and keyword search
- **Vision Support**: GPT-4o integration for image description
- **Source Attribution**: Track document sources in answers
- **REST API**: FastAPI endpoints for ingestion and querying

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Endpoints                        │
├─────────────────────────────────────────────────────────────────┤
│  POST /ingest/file  │  POST /query  │  GET /stats  │  ...      │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Processors    │  │    Retrieval    │  │   Generation    │
│  PDF, DOCX, ...│  │  KNN + Hybrid   │  │  LangChain LLM  │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    Chunking     │  │   Embeddings    │  │    OpenAI API   │
│  Smart Split    │  │ Sentence-Trans. │  │   GPT-4o-mini   │
└────────┬────────┘  └────────┬────────┘  └─────────────────┘
         │                    │
         └────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │     OpenSearch      │
         │   Vector Database   │
         └─────────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key

### 1. Clone and Configure

```bash
# Clone repository
git clone <your-repo-url>
cd multimodal-rag

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Start Services

```bash
# Start OpenSearch and API
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Check supported formats
curl http://localhost:8000/supported-formats

# Upload a document
curl -X POST "http://localhost:8000/ingest/file" \
  -F "file=@document.pdf"

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "top_k": 5}'
```

## API Endpoints

### Ingestion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest/file` | POST | Upload and ingest a single file |
| `/ingest/batch` | POST | Upload multiple files |
| `/ingest/texts` | POST | Ingest raw text strings |

### Query

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | RAG query with answer generation |
| `/retrieve` | GET | Retrieve documents without generation |

### Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stats` | GET | Document statistics |
| `/documents/{source}` | DELETE | Delete documents by source |
| `/clear` | POST | Clear all documents |

## Supported File Formats

| Format | Extensions | Processor |
|--------|------------|-----------|
| PDF | `.pdf` | pymupdf4llm (markdown extraction) |
| Word | `.docx`, `.doc` | python-docx |
| Excel | `.xlsx`, `.xls`, `.csv` | openpyxl + pandas |
| PowerPoint | `.pptx` | python-pptx |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | GPT-4o Vision |
| Text | `.txt`, `.md` | Direct read |

## Configuration

Environment variables (`.env`):

```bash
# OpenSearch
OPENSEARCH_HOST=http://localhost:9200
INDEX_NAME=multimodal_docs

# Embeddings
EMBED_MODEL=all-MiniLM-L6-v2
EMBED_DIM=384

# LLM
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini
VISION_MODEL=gpt-4o

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Processing
OCR_ENABLED=true
VISION_DESCRIPTION_ENABLED=true
```

## Project Structure

```
multimodal-rag/
├── app/
│   ├── processors/          # Document processors
│   │   ├── base.py          # Base class & Document schema
│   │   ├── pdf_processor.py
│   │   ├── docx_processor.py
│   │   ├── xlsx_processor.py
│   │   ├── pptx_processor.py
│   │   ├── image_processor.py
│   │   ├── text_processor.py
│   │   └── factory.py       # Auto-detection
│   ├── chunking/            # Text splitting
│   │   └── chunker.py
│   ├── embeddings/          # Vector generation
│   │   └── embedding.py
│   ├── opensearch/          # Vector database
│   │   └── opensearch_client.py
│   ├── retrieval/           # Search & retrieval
│   │   └── retriever.py
│   ├── generation/          # LLM generation
│   │   └── generator.py
│   ├── ingestion/           # Ingestion pipeline
│   │   └── ingest.py
│   ├── config.py            # Configuration
│   └── main.py              # FastAPI app
├── tests/                   # Test suite
├── notebooks/               # Jupyter notebooks
├── data/uploads/            # Uploaded files
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Development

### Local Setup (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start OpenSearch (Docker)
docker run -d -p 9200:9200 -e "discovery.type=single-node" \
  -e "plugins.security.disabled=true" \
  opensearchproject/opensearch:2.12.0

# Run API
uvicorn app.main:app --reload
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_processors.py -v
```

### Using Jupyter Notebook

```bash
# Start notebook service
docker-compose up notebook

# Open http://localhost:8888
# Navigate to work/Multimodal_RAG_Workflow.ipynb
```

## Usage Examples

### Python SDK

```python
import httpx

API_URL = "http://localhost:8000"

# Upload a file
with open("document.pdf", "rb") as f:
    response = httpx.post(
        f"{API_URL}/ingest/file",
        files={"file": f}
    )
print(response.json())

# Query
response = httpx.post(
    f"{API_URL}/query",
    json={
        "question": "What are the key findings?",
        "top_k": 5,
        "file_type": "pdf"  # Optional filter
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### Direct Module Usage

```python
from app.processors import process_file
from app.chunking import chunk_documents
from app.embeddings import embed_async
from app.opensearch import bulk_insert
from app.retrieval import retrieve_top_k
from app.generation import generate_answer

# Process a file
docs = process_file("report.pdf")

# Chunk
chunks = chunk_documents(docs)

# Embed and store
vectors = await embed_async([c.content for c in chunks])
for chunk, vec in zip(chunks, vectors):
    chunk.embedding = vec.tolist()
bulk_insert([c.to_dict() for c in chunks])

# Query
results = await retrieve_top_k("What is the summary?", top_k=5)
answer = await generate_answer(
    "What is the summary?",
    [r["content"] for r in results]
)
```

## Performance Tips

1. **Batch Processing**: Use `/ingest/batch` for multiple files
2. **Chunk Size**: Adjust `CHUNK_SIZE` based on your documents (500-2000 chars)
3. **Top-K**: Use lower `top_k` (3-5) for focused answers
4. **Filters**: Use `file_type` filter to narrow search scope
5. **Hybrid Search**: Enable for better keyword matching

## Troubleshooting

### OpenSearch Connection Failed
```bash
# Check if OpenSearch is running
curl http://localhost:9200

# Check Docker logs
docker-compose logs opensearch
```

### Out of Memory
```bash
# Increase OpenSearch memory in docker-compose.yml
OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g
```

### Embedding Model Download
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
