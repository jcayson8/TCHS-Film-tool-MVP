"""Response models exposed by the defensive AI service."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    service: str
    status: str
    version: str
    python_version: str
    model_connected: bool


class ClassDefinition(BaseModel):
    index: int
    name: str


class ClassesResponse(BaseModel):
    classes: list[ClassDefinition]


class ConfigResponse(BaseModel):
    service: str
    version: str
    dataset: str
    class_count: int
    object_classes: list[ClassDefinition]
    derived_football_labels: list[str]
