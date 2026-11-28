"""FastAPI application entry point for CSRA."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

import uvicorn
from fastapi import FastAPI

from csra import __version__
from csra.api.routes import router as support_router
from csra.utils.config import get_settings
from csra.utils.logging import setup_logging

settings = get_settings()
setup_logging(settings.log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of shared application resources."""
    app.state.llm_client = None
    yield
    client = getattr(app.state, "llm_client", None)
    if client is not None:
        await client.aclose()


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Customer Support Resolution Agent (CSRA) API",
    lifespan=lifespan,
)

app.include_router(support_router)


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    """Return a simple response indicating that CSRA is running."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "status": "running",
    }


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Return basic application health information."""
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "version": __version__,
    }


if __name__ == "__main__":
    logger.info("Starting %s in %s mode", settings.app_name, settings.app_env)
    uvicorn.run(
        "csra.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
    )
