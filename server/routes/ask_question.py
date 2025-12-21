from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import pinecone

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings

from logger import logger
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain

router = APIRouter()

# -------------------------
# Request model
# -------------------------
class AskRequest(BaseModel):
    query: str


# -------------------------
# Pinecone v8 client
# -------------------------
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])


@router.post("/ask")
async def ask_question(data: AskRequest):
    try:
        question = data.query
        logger.info(f"User query: {question}")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        query_vector = embeddings.embed_query(question)

        response = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in response.get("matches", [])
        ]

        if not docs:
            return {
                "answer": "No relevant information found in uploaded documents.",
                "sources": []
            }

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

        logger.info("RAG query successful")
        return result

    except Exception as e:
        logger.exception("RAG query failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
