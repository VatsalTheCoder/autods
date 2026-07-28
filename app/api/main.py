"""FastAPI application.

Section 0 exposes only /health. Upload and job endpoints arrive in Sections 1-4,
and the results endpoints the Results page reads in Section 5.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.routes import chat, results, upload
from app.core.config import get_settings
from app.core.db import database_healthy
from app.core.storage import storage_healthy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="AutoDS API",
    description="Multi-agent autonomous data scientist",
    version="0.1.0",
    debug=settings.debug,
)


app.include_router(upload.router)
app.include_router(results.router)
app.include_router(chat.router)


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str
    dependencies: dict[str, bool]


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> JSONResponse:
    """Liveness and dependency check.

    Returns 200 only when both Postgres and object storage are reachable, and
    503 otherwise, so a load balancer pulls the instance out of rotation rather
    than sending traffic to a container that cannot do any real work (spec
    section 14). The per-dependency breakdown is what makes this useful for
    debugging: it tells you *which* service is down, not just that one is.
    """
    dependencies = {
        "database": database_healthy(),
        "storage": storage_healthy(),
    }
    all_up = all(dependencies.values())

    body = HealthResponse(
        status="healthy" if all_up else "degraded",
        environment=settings.environment,
        version=app.version,
        dependencies=dependencies,
    )
    code = status.HTTP_200_OK if all_up else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=body.model_dump())


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"service": "AutoDS API", "docs": "/docs", "health": "/health"}
