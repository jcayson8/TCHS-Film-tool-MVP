"""FastAPI application entry point."""

from fastapi import FastAPI

from app.config import SERVICE_NAME, SERVICE_VERSION
from app.routes import classes, detection, health, tracking

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)
app.include_router(health.router)
app.include_router(classes.router)
app.include_router(detection.router)
app.include_router(tracking.router)
