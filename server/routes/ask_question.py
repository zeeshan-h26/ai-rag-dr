from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import os

from pydantic import Field
from pinecone import Pinecone
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# HF Endpoint Embeddings (correct new class)
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Correct project imports
from server.modules.llm import get_llm_chain
from server.modules.query_handlers import query_chain
from server.logger import logger


router = APIRouter()

# -------------------------
# Load Environment
# -------------------------
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN missing")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME missing")


# -------------------------
# Global Initialization
# -------------------------

# HuggingFace BGE-M3 Embeddings (correct + stable)
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3",
    
)

# Pinecone v3 client + index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# -------------------------
# Ask Endpoint
# -------------------------
@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"User query: {question}")

        # -------------------------
        # Embed Query  (BGE requires "query:" prefix)
        # -------------------------
        embedded_query = embeddings.embed_query(f"query: {question}")

        # -------------------------
        # Query Pinecone
        # -------------------------
        response = index.query(
            vector=embedded_query,
            top_k=3,
            include_metadata=True
        )

        matches = response.get("matches", [])

        docs = [
            Document(
                page_content=m["metadata"].get("text", ""),
                metadata=m["metadata"]
            )
            for m in matches
        ]

        if not docs:
            return {
                "answer": "Sorry, no relevant information found in uploaded documents.",
                "sources": []
            }

        # -------------------------
        # Simple Retriever
        # -------------------------
        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)

        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("Query processed successfully")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
