from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ✅ Use package imports
from server.middlewares.exception_handlers import catch_exception_middleware
from server.routes.upload_pdfs import router as upload_router
from server.routes.ask_question import router as ask_router   # make sure file name is ask_questions.py

app = FastAPI(
    title="Medical Assistant API",
    description="API for AI Medical Assistant Chatbot"
)

# ----------------------------
# CORS Setup
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------------------
# Middleware
# ----------------------------
app.middleware("http")(catch_exception_middleware)

# ----------------------------
# Routers
# ----------------------------
app.include_router(upload_router)
app.include_router(ask_router)
