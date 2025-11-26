# FILE: app/generation/generator.py
"""
RAG Answer Generation using LangChain.
Generates final answers from retrieved documents.
"""

import os
from typing import List, Dict, Any, Optional
import logging

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


# =============================================================================
# RAG Prompt Templates
# =============================================================================

RAG_PROMPT = PromptTemplate.from_template("""
You are a highly skilled assistant with expertise in analyzing documents.
Answer the question using ONLY the retrieved documents below.
If the answer is not in the documents, say "I cannot answer this from the retrieved data."

DOCUMENTS:
{context}

QUESTION:
{question}

Provide a clear, accurate answer based on the documents. If relevant, cite which document or source the information comes from.
""")


RAG_PROMPT_WITH_SOURCES = PromptTemplate.from_template("""
You are a highly skilled assistant with expertise in analyzing documents.
Answer the question using ONLY the retrieved documents below.
If the answer is not in the documents, say "I cannot answer this from the retrieved data."

DOCUMENTS:
{context}

SOURCES:
{sources}

QUESTION:
{question}

Provide a clear, accurate answer. Cite the source(s) for key information using the format [Source: filename].
""")


FORMULATION_PROMPT = PromptTemplate.from_template("""
You are a highly skilled formulation scientist.
Answer the question using ONLY the retrieved documents below.
If the answer is not in the documents, say "I cannot answer this from the retrieved data."

DOCUMENTS:
{context}

QUESTION:
{question}

Answer concisely and scientifically. Include relevant formulation details, percentages, and technical specifications when available.
""")


# =============================================================================
# LLM Initialization
# =============================================================================

def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """
    Initialize the ChatOpenAI LLM.
    
    Parameters
    ----------
    model : Optional[str]
        Model name override.
    temperature : float
        Generation temperature.
    api_key : Optional[str]
        API key override.
        
    Returns
    -------
    ChatOpenAI
        Configured LLM instance.
    """
    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", LLM_MODEL),
        temperature=temperature,
        api_key=api_key or os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
    )


# =============================================================================
# Generation Functions
# =============================================================================

async def generate_answer(
    question: str,
    retrieved_docs: List[str],
    prompt_template: Optional[PromptTemplate] = None,
) -> str:
    """
    Generate a RAG answer from retrieved documents.
    
    Parameters
    ----------
    question : str
        User question.
    retrieved_docs : List[str]
        Retrieved document contents.
    prompt_template : Optional[PromptTemplate]
        Custom prompt template.
        
    Returns
    -------
    str
        Generated answer.
    """
    if not retrieved_docs:
        return "No relevant documents retrieved — cannot answer."
    
    # Combine documents
    context = "\n\n---\n\n".join(
        f"[Document {i+1}]\n{doc}" 
        for i, doc in enumerate(retrieved_docs)
    )
    
    # Use provided or default prompt
    prompt = prompt_template or RAG_PROMPT
    
    # Initialize LLM and chain
    llm = get_llm()
    chain = prompt | llm
    
    # Generate answer
    try:
        ai_message = await chain.ainvoke({
            "context": context,
            "question": question,
        })
        return ai_message.content.strip()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error generating answer: {str(e)}"


async def generate_answer_with_sources(
    question: str,
    documents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate answer with source citations.
    
    Parameters
    ----------
    question : str
        User question.
    documents : List[Dict[str, Any]]
        Retrieved documents with content and source info.
        
    Returns
    -------
    Dict[str, Any]
        Answer with sources.
    """
    if not documents:
        return {
            "answer": "No relevant documents retrieved — cannot answer.",
            "sources": [],
        }
    
    # Format context and sources
    context_parts = []
    source_list = []
    
    for i, doc in enumerate(documents):
        content = doc.get("content", "")
        source = doc.get("source", f"Document {i+1}")
        
        context_parts.append(f"[{source}]\n{content}")
        source_list.append(source)
    
    context = "\n\n---\n\n".join(context_parts)
    sources_str = "\n".join(f"- {s}" for s in source_list)
    
    # Generate
    llm = get_llm()
    chain = RAG_PROMPT_WITH_SOURCES | llm
    
    try:
        ai_message = await chain.ainvoke({
            "context": context,
            "sources": sources_str,
            "question": question,
        })
        
        return {
            "answer": ai_message.content.strip(),
            "sources": list(set(source_list)),  # Unique sources
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
        }


async def generate_formulation_answer(
    question: str,
    retrieved_docs: List[str],
) -> str:
    """
    Generate answer using formulation-specific prompt.
    
    Parameters
    ----------
    question : str
        Formulation-related question.
    retrieved_docs : List[str]
        Retrieved documents.
        
    Returns
    -------
    str
        Scientific answer.
    """
    return await generate_answer(
        question,
        retrieved_docs,
        prompt_template=FORMULATION_PROMPT,
    )


class RAGGenerator:
    """
    RAG Generator class for configurable generation.
    """
    
    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = 0.0,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        """
        Initialize generator.
        
        Parameters
        ----------
        model : str
            LLM model name.
        temperature : float
            Generation temperature.
        prompt_template : Optional[PromptTemplate]
            Custom prompt.
        """
        self.model = model
        self.temperature = temperature
        self.prompt_template = prompt_template or RAG_PROMPT
        self._llm = None
    
    @property
    def llm(self) -> ChatOpenAI:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = get_llm(
                model=self.model,
                temperature=self.temperature,
            )
        return self._llm
    
    async def generate(
        self,
        question: str,
        documents: List[str],
    ) -> str:
        """
        Generate answer.
        
        Parameters
        ----------
        question : str
            User question.
        documents : List[str]
            Retrieved documents.
            
        Returns
        -------
        str
            Generated answer.
        """
        if not documents:
            return "No relevant documents retrieved — cannot answer."
        
        context = "\n\n---\n\n".join(
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(documents)
        )
        
        chain = self.prompt_template | self.llm
        
        ai_message = await chain.ainvoke({
            "context": context,
            "question": question,
        })
        
        return ai_message.content.strip()
