# AI-ML Orchestration CPG

A comprehensive machine learning orchestration platform designed for the Consumer Packaged Goods (CPG) industry, featuring advanced formulation prediction, product performance analysis, and generative AI capabilities.

## Overview

This repository contains multiple machine learning projects and modules that work together to provide end-to-end ML orchestration for CPG applications, including:

- **Advanced Formulation** - ML models for advanced product formulations
- **Cosmetics Augmented Formulation** - Augmented formulation models for cosmetics with microbiology and stability predictions
- **Cosmetics Generative Formulation** - Generative AI models for beauty/personal care and food/beverage formulations
- **Orchestration Properties Prediction** - Dairy properties prediction with multiple ONNX models for production inference
- **Product Performance Prediction** - End-to-end ML pipeline for predicting product performance metrics
- **Multimodal RAG** - Retrieval-Augmented Generation system with Docker containerization
- **PDF Extraction with OCR & GPT** - Extract materials information from documents using OCR and GPT
- **Retrieval-Augmented Generation Scientific Data** - RAG system for scientific data processing

## Projects

### ml-advanced-formulation
Advanced machine learning models for formulation development with Jupyter notebooks and modular source code.

### ml-cosmetics-augmented-formulation
Specialized models for cosmetics with:
- Microbiology inference and training
- Stability prediction models
- Integrated data pipeline

### ml-cosmetics-generative-formulation
Generative AI models organized by category:
- **beauty-personal-care** - Beauty and personal care product formulations
- **food-beverage** - Food and beverage formulation generation

### ml-orchestration-properties-prediction
Production-ready ML orchestration with:
- Multiple pre-trained ONNX models for dairy properties
- Comprehensive model metrics and validation
- Real-time inference capabilities

### ml-product-performance-prediction
Full ML pipeline including:
- Data processing and feature engineering
- Model training and validation
- Performance prediction outputs
- Comprehensive documentation

### multimodal-rag
Docker-based Retrieval-Augmented Generation system with:
- Multi-modal data processing
- Docker and Docker Compose setup
- Python package management with pyproject.toml

### pdf-extraction-with-ocr-gpt
Document processing with:
- OCR capabilities
- GPT-based material extraction
- Support for 3DS document format

### retrieval-augmented-generation-scientific-data
Scientific data RAG system featuring:
- Docker containerization
- Advanced RAG workflows
- Scientific document processing

## Getting started

To get started with this repository:

1. **Clone the repository**
   ```bash
   git clone https://gitlab.paris.exalead.com/MJI11/ai-ml-orchestration-cpg.git
   cd ai-ml-orchestration-cpg
   ```

2. **Explore the projects**
   - Each subdirectory contains a complete project with its own README
   - Start with the project that matches your use case

3. **Set up your environment**
   - Most projects include a `requirements.txt` or `pyproject.toml`
   - Docker support is available for containerized projects
   - Jupyter notebooks are provided for exploratory analysis

4. **Review project-specific documentation**
   - Navigate to individual project directories for detailed setup instructions
   - Each project contains its own README with specific requirements and usage

## Key Features

- **Production-Ready Models** - ONNX-formatted models for inference at scale
- **Multi-Project Structure** - Modular organization for different ML use cases
- **Docker Support** - Containerized deployments for cloud-native applications
- **Jupyter Notebooks** - Interactive exploratory analysis and model development
- **Generative AI** - Advanced generative models for formulation design
- **RAG Systems** - Retrieval-augmented generation for scientific data
- **Data Processing** - Comprehensive data pipelines and feature engineering

## Technologies

- Python 3.x
- Machine Learning: Scikit-Learn, LightGBM, RandomForest, Gradient Boosting
- Deep Learning frameworks as needed per project
- Docker & Docker Compose
- Jupyter Notebooks
- ONNX for model serialization
- RAG frameworks
- OCR and GPT integration

## Support

For issues or questions about specific projects, refer to the README in each project directory.
