"""FastAPI application entrypoint with CORS and route mounting."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routes import router
from src.configs.settings import settings

app = FastAPI(title="RAG Chat API (No LangChain)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/healthz")
def healthz():
    """Returns a simple health status payload for readiness checks."""
    return {"status": "ok"}
