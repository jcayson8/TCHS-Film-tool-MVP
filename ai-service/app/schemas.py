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


class ModelStatusResponse(BaseModel):
    service: str
    vision_dependencies_available: bool
    model_configured: bool
    model_loaded: bool
    model_name: str
    model_mode: str | None
    device: str | None
    allow_model_download: bool
    last_error: str | None
    supported_generic_classes: list[str]
    locked_defensive_classes: list[ClassDefinition]


class Detection(BaseModel):
    id: str
    source_class_index: int
    source_class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    suggested_class_index: int | None
    suggested_class_name: str | None
    needs_classification: bool
    source: str
    model: str


class DetectionResponse(BaseModel):
    frame_width: int
    frame_height: int
    model: str
    model_mode: str
    device: str
    inference_time_ms: float
    detection_count: int
    detections: list[Detection]


class TrackedAnnotation(BaseModel):
    player_id: str
    class_index: int
    x: float
    y: float
    width: float
    height: float
    tracking_confidence: str
    confidence_score: float


class TrackedFrame(BaseModel):
    frame_index: int
    video_frame_number: int
    frame_time_ms: int
    annotations: list[TrackedAnnotation]


class TrackingFailure(BaseModel):
    player_id: str
    frame_index: int
    video_frame_number: int
    frame_time_ms: int
    reason: str


class TrackingResponse(BaseModel):
    tracker: str
    requested_frame_count: int
    completed_frame_count: int
    frames: list[TrackedFrame]
    failures: list[TrackingFailure]
