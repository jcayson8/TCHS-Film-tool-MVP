"""FastAPI application entry point."""

from fastapi import FastAPI

from app.config import SERVICE_NAME, SERVICE_VERSION
from app.routes import classes, health

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)
app.include_router(health.router)
app.include_router(classes.router)
