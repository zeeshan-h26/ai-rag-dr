
from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from server.middlewares.exception_handlers import catch_exception_middleware
from server.routes.upload_pdfs import router as upload_router
from server.routes.ask_question import router as ask_router


app = FastAPI(
    title="Medical Assistant API",
    description="RAG system using LangChain + Pinecone"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(catch_exception_middleware)

app.include_router(upload_router)
app.include_router(ask_router)

@app.get("/")
def health():
    return {"status": "Backend running"}
