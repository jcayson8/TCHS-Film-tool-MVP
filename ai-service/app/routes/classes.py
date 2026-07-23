"""Defensive class and public configuration routes."""

from fastapi import APIRouter

from app.config import (
    DATASET_NAME,
    DERIVED_FOOTBALL_LABELS,
    OBJECT_CLASSES,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from app.schemas import ClassDefinition, ClassesResponse, ConfigResponse

router = APIRouter(tags=["configuration"])


def _object_class_definitions() -> list[ClassDefinition]:
    return [
        ClassDefinition(index=index, name=name)
        for index, name in enumerate(OBJECT_CLASSES)
    ]


@router.get("/classes", response_model=ClassesResponse)
def get_classes() -> ClassesResponse:
    """Return the locked V1 object classes in index order."""
    return ClassesResponse(classes=_object_class_definitions())


@router.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    """Return non-secret model-foundation configuration."""
    return ConfigResponse(
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        dataset=DATASET_NAME,
        class_count=len(OBJECT_CLASSES),
        object_classes=_object_class_definitions(),
        derived_football_labels=list(DERIVED_FOOTBALL_LABELS),
    )
