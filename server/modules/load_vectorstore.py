import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm

# Pinecone v3
from pinecone import Pinecone

# LangChain loaders & splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HuggingFace Endpoint Embeddings (Correct + Stable)
from langchain_huggingface import HuggingFaceEndpointEmbeddings


# ----------------------------------------
# Load ENV
# ----------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME missing")

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------------------------------------
# Connect Pinecone
# ----------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# ----------------------------------------
# Load, split, embed, upload
# ----------------------------------------
def load_vectorstore(uploaded_files):

    # BGE-M3 via HF Endpoint API (1024-dim)
    embedder = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3"
        # IMPORTANT: Do NOT pass token here.
        # It auto reads from HUGGINGFACEHUB_API_TOKEN environment variable
    )

    file_paths = []

    # Save uploaded PDFs
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    # Process each PDF
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120
        )
        chunks = splitter.split_documents(documents)

        # BGE requires "passage:" prefix
        texts = [f"passage: {chunk.page_content}" for chunk in chunks]

        metadatas = [
            {
                "text": chunk.page_content,  # required for RAG response
                **chunk.metadata
            }
            for chunk in chunks
        ]

        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embedder.embed_documents(texts)

        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone"):
            index.upsert(vectors=zip(ids, embeddings, metadatas))

        print(f"✅ Upload complete → {file_path}")
