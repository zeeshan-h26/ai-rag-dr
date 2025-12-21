from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os

from pinecone import Pinecone

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from logger import logger

router = APIRouter(prefix="/upload", tags=["Upload"])

# =========================
# Pinecone v3 setup
# =========================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in environment")

if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME not found in environment")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# =========================
# Upload endpoint
# =========================

@router.post("/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        logger.info("Uploading document")

        text = (await file.read()).decode("utf-8", errors="ignore")

        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector = embedder.embed_query(text)

        index.upsert(
            vectors=[
                {
                    "id": file.filename,
                    "values": vector,
                    "metadata": {"text": text}
                }
            ]
        )

        return {"message": "Document uploaded successfully"}

    except Exception as e:
        logger.exception("Upload failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
