import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm

# Pinecone (latest SDK)
from pinecone import Pinecone

# LangChain loaders & splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ HuggingFace embeddings (FREE, no API key)
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ----------------------------------------
# ENV variables
# ----------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ YOUR EXISTING INDEX NAME
PINECONE_INDEX_NAME = "aidoctor"

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------------------
# Connect to EXISTING Pinecone index
# ----------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ----------------------------------------
# Load, split, embed, and upsert PDFs
# ----------------------------------------
def load_vectorstore(uploaded_files):
    # ✅ HuggingFace embedding model (384-dim)
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
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
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # ✅ Extract text + store text explicitly in metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "text": chunk.page_content,  # IMPORTANT for RAG
                **chunk.metadata
            }
            for chunk in chunks
        ]

        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids, embeddings, metadatas))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")
