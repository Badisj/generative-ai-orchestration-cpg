"""
generator.py
------------------------
Responsible for generating final answers from retrieved documents
using the newest LangChain 2024+ API.

Features:
- Async pipeline
- Uses PromptTemplate from langchain_core
- Uses ChatOpenAI from langchain_openai
- Works seamlessly with your existing retrieval pipeline
"""

import os
from typing import List

from langchain.prompts import PromptTemplate
# from langchain.output_parsers import TextOutputParser
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY


# -----------------------------------------------------------------------------
# RAG Prompt Template
# -----------------------------------------------------------------------------
RAG_PROMPT = PromptTemplate.from_template("""
You are a highly skilled formulation scientist.
Answer the question using ONLY the retrieved documents below.
If the answer is not in the documents, say "I cannot answer this from the retrieved data."

DOCUMENTS:
{context}

QUESTION:
{question}

Answer concisely and scientifically.
""")


# -----------------------------------------------------------------------------
# Initialize LLM
# -----------------------------------------------------------------------------
def get_llm() -> ChatOpenAI:
    """
    Initialize the ChatOpenAI LLM using environment variable LLM_MODEL.
    """
    api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0.0, api_key=api_key)


# -----------------------------------------------------------------------------
# Async RAG Answer Generation
# -----------------------------------------------------------------------------
async def generate_answer(question: str, retrieved_docs: List[str]) -> str:
    """
    Generate a final answer from retrieved documents using LangChain 2024+.

    Parameters
    ----------
    question : str
        User question to answer.

    retrieved_docs : List[str]
        Retrieved documents from OpenSearch.

    Returns
    -------
    str
        Generated answer text.
    """
    if not retrieved_docs:
        return "No relevant documents retrieved — cannot answer."

    # Combine documents into context
    context = "\n\n".join(retrieved_docs)

    # Initialize LLM
    llm = get_llm()
    # parser = TextOutputParser()

    # Pipe-style chain: Prompt -> LLM -> Parser
    # chain = RAG_PROMPT | llm | parser

    # Pipe-style chain: Prompt -> LLM  
    chain = RAG_PROMPT | llm


    # Execute async
    ai_message = await chain.ainvoke({
        "context": context,
        "question": question
    })

    # Extract content from AIMessage
    return ai_message.content.strip()
