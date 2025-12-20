from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import os

from pydantic import Field
from pinecone import Pinecone

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# ✅ HuggingFace embeddings (FREE)
from langchain_community.embeddings import HuggingFaceEmbeddings

from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger

router = APIRouter()


@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"User query: {question}")

        # --------------------------------
        # Pinecone connection
        # --------------------------------
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

        # --------------------------------
        # HuggingFace Embedding model
        # MUST match the one used during upsert
        # --------------------------------
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        embedded_query = embed_model.embed_query(question)

        # --------------------------------
        # Query Pinecone
        # --------------------------------
        res = index.query(
            vector=embedded_query,
            top_k=3,
            include_metadata=True
        )

        # --------------------------------
        # Convert Pinecone matches → LangChain Documents
        # Uses "text" stored in metadata
        # --------------------------------
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in res.get("matches", [])
        ]

        if not docs:
            return {
                "answer": "I'm sorry, I couldn't find relevant information in the uploaded documents.",
                "sources": []
            }

        # --------------------------------
        # Simple Retriever (LangChain 1.x)
        # --------------------------------
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

        # --------------------------------
        # LLM Chain + Query
        # --------------------------------
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
