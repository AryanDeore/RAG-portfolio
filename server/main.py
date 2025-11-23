"""FastAPI application entrypoint with CORS and route mounting."""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routes import router
from src.configs.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Configure LiteLLM to use OpenRouter
# LiteLLM will use OPENROUTER_API_KEY environment variable if set
# For OpenRouter, models should be prefixed with 'openrouter/' or use the base model name
# with api_base set to OpenRouter's endpoint
try:
    import litellm
    # Set OpenRouter API key if available
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
        # Configure LiteLLM to use OpenRouter for OpenAI models
        # This routes all openai/* models through OpenRouter
        litellm.api_key = openrouter_key
        logger = logging.getLogger(__name__)
        logger.info("LiteLLM configured to use OpenRouter")
    else:
        logger = logging.getLogger(__name__)
        logger.warning("OPENROUTER_API_KEY not set. LiteLLM may not work correctly.")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"LiteLLM configuration failed: {e}")

# Initialize Opik for tracing
# Opik will use environment variables: OPIK_API_KEY and OPIK_WORKSPACE
# If not set, it will use local mode
try:
    import opik
    opik.configure()
    logger = logging.getLogger(__name__)
    logger.info("Opik tracing initialized")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Opik initialization failed: {e}. Tracing will be disabled.")

app = FastAPI(title="Aryan's Portfolio Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    """Root endpoint providing API information."""
    return {
        "service": "Aryan's Portfolio Assistant",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "health": "/healthz",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/healthz")
def healthz():
    """Returns a simple health status payload for readiness checks."""
    return {"ok": True, "service": "Aryan's Portfolio Assistant", "docs": "/docs", "health": "/healthz"}