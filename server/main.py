"""FastAPI application entrypoint with CORS and route mounting."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routes import router
from src.configs.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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
    return {"ok": True, "service": "RAG Chat API", "docs": "/docs", "health": "/healthz"}
