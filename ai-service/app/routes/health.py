"""Service health route."""

import platform

from fastapi import APIRouter

from app.config import SERVICE_NAME, SERVICE_VERSION
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report service availability without loading a model."""
    return HealthResponse(
        service=SERVICE_NAME,
        status="ok",
        version=SERVICE_VERSION,
        python_version=platform.python_version(),
        model_connected=False,
    )
