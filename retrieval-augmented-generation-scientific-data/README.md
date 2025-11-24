# RAG Service for Scientific Formulations


## Objectives
- Implement a production-ready Retrieval-Augmented Generation (RAG) backend.
- Use open-source components for embeddings and vector storage.
- Enable fast retrieval of scientific formulation and raw material data.
- Provide async API endpoints for ingestion and query.


## Architecture
- **FastAPI**: Async backend serving endpoints.
- **OpenSearch**: Open-source vector database with KNN search.
- **Sentence-Transformers**: Async embedding generation.
- **LangChain / OSS LLM**: Generates answers from retrieved documents.
- **Ingestion pipeline**: Async insertion of raw text.
- **Tests**: Async smoke tests for retrieval and ingestion.
- **Docker Compose**: Local OpenSearch + API.
- **CI/CD**: GitHub Actions workflow to run tests.


## File Structure & Description


### app/main.py
Entry point for FastAPI application. Mounts API routes and health endpoint.


### app/config.py
Contains configuration variables such as OpenSearch host, index name, embedding model.


### app/embeddings/embedding.py
Async wrapper around Sentence-Transformers embedding model. Converts text into vector embeddings.


### app/opensearch/opensearch_client.py
OpenSearch client wrapper with bulk insertion and KNN search. Ensures index exists.


### app/retrieval/retriever.py
Retrieves top-k relevant documents by embedding the query and using OpenSearch KNN.


### app/generation/generator.py
Uses LangChain or OSS LLM to generate an answer based on retrieved context.


### app/ingestion/ingest.py
Async ingestion module to embed and insert text documents into OpenSearch.


### app/api/routes.py
Defines API endpoints for insertion (`/rag/insert`) and query (`/rag/query`).


### tests/test_rag.py
Async smoke tests to validate the retriever component.


### docker-compose.yml
Launches OpenSearch container and API container for local development.


### .github/workflows/ci.yml
CI/CD workflow to install dependencies and run tests on push and pull requests.