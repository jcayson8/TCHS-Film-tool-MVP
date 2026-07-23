"""Optional current-frame detection routes."""

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import TypeAdapter

from app.config import settings
from app.schemas import DetectionResponse, ModelStatusResponse
from app.services import detector

router = APIRouter(tags=["detection"])


@router.get("/model-status", response_model=ModelStatusResponse)
def model_status() -> dict:
    return detector.status()


@router.post("/detect/frame", response_model=DetectionResponse)
async def detect_frame(
    image: Annotated[UploadFile, File()],
    confidence: Annotated[float | None, Form(ge=0.01, le=1.0)] = None,
    iou: Annotated[float | None, Form(ge=0.01, le=1.0)] = None,
) -> dict:
    data = await image.read(detector.MAX_IMAGE_BYTES + 1)
    confidence_value = TypeAdapter(float).validate_python(
        confidence if confidence is not None else settings.detector_confidence
    )
    iou_value = TypeAdapter(float).validate_python(
        iou if iou is not None else settings.detector_iou
    )
    return detector.detect(
        data,
        image.content_type or "",
        confidence_value,
        iou_value,
    )
