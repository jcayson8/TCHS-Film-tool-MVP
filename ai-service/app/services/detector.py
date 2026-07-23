"""Lazy, optional YOLO inference for a single uploaded frame."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from app.config import AI_SERVICE_DIR, OBJECT_CLASSES, SERVICE_NAME, settings

GENERIC_CLASSES = ("person", "sports ball")
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
MAX_IMAGE_PIXELS = 40_000_000

_load_lock = Lock()
_model: Any | None = None
_torch: Any | None = None
_device: str | None = None
_model_mode: str | None = None
_last_error: str | None = None


def _safe_model_name() -> str:
    return Path(settings.detector_model).name


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from PIL import Image, UnidentifiedImageError
        from ultralytics import YOLO
    except (ImportError, ModuleNotFoundError) as error:
        raise HTTPException(
            status_code=503,
            detail="Vision dependencies are unavailable. Install requirements-vision.txt.",
        ) from error
    return torch, Image, (YOLO, UnidentifiedImageError)


def _configured_model_reference() -> str:
    configured = Path(settings.detector_model).expanduser()
    candidates = [configured] if configured.is_absolute() else [
        AI_SERVICE_DIR / configured,
        settings.model_dir / configured,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    if not settings.allow_model_download:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Detector weights '{_safe_model_name()}' are unavailable locally. "
                "Set AI_ALLOW_MODEL_DOWNLOAD=true to permit Ultralytics download resolution."
            ),
        )
    return settings.detector_model


def _select_device(torch: Any) -> str:
    configured = settings.detector_device.strip().lower()
    if configured != "auto":
        return configured
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, list):
        return dict(enumerate(str(name) for name in names))
    return {int(index): str(name) for index, name in names.items()}


def _mode_for_names(names: dict[int, str]) -> str:
    values = set(names.values())
    return (
        "custom_defensive_classes"
        if set(OBJECT_CLASSES).issubset(values)
        else "generic_person_ball"
    )


def get_model() -> tuple[Any, str, str]:
    global _model, _torch, _device, _model_mode, _last_error
    if _model is not None:
        return _model, _device or "cpu", _model_mode or "generic_person_ball"
    with _load_lock:
        if _model is not None:
            return _model, _device or "cpu", _model_mode or "generic_person_ball"
        try:
            torch, _image, dependency = _dependencies()
            YOLO, _unidentified = dependency
            model = YOLO(_configured_model_reference())
            _torch = torch
            _device = _select_device(torch)
            _model_mode = _mode_for_names(_class_names(model))
            _model = model
            _last_error = None
        except HTTPException as error:
            _last_error = str(error.detail)
            raise
        except Exception as error:
            _last_error = "Detector model initialization failed."
            raise HTTPException(status_code=503, detail=_last_error) from error
    return _model, _device or "cpu", _model_mode or "generic_person_ball"


def status() -> dict[str, Any]:
    try:
        _dependencies()
        dependencies_available = True
    except HTTPException:
        dependencies_available = False
    status_error = _last_error
    if dependencies_available and _model is None and not settings.allow_model_download:
        configured = Path(settings.detector_model).expanduser()
        candidates = [configured] if configured.is_absolute() else [
            AI_SERVICE_DIR / configured,
            settings.model_dir / configured,
        ]
        if not any(candidate.is_file() for candidate in candidates):
            status_error = f"Detector weights '{_safe_model_name()}' are unavailable locally."
    return {
        "service": SERVICE_NAME,
        "vision_dependencies_available": dependencies_available,
        "model_configured": bool(settings.detector_model),
        "model_loaded": _model is not None,
        "model_name": _safe_model_name(),
        "model_mode": _model_mode,
        "device": _device,
        "allow_model_download": settings.allow_model_download,
        "last_error": status_error,
        "supported_generic_classes": list(GENERIC_CLASSES),
        "locked_defensive_classes": [
            {"index": index, "name": name}
            for index, name in enumerate(OBJECT_CLASSES)
        ],
    }


def decode_image(data: bytes, content_type: str) -> Any:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds 12 MB.")
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if content_type.lower() not in allowed:
        raise HTTPException(status_code=415, detail="Use a JPEG, PNG, or WebP image.")
    _torch_dep, Image, dependency = _dependencies()
    _YOLO, UnidentifiedImageError = dependency
    try:
        image = Image.open(BytesIO(data))
        image.verify()
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Uploaded image could not be decoded.") from error
    width, height = image.size
    if (
        width <= 0 or height <= 0
        or width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise HTTPException(status_code=400, detail="Uploaded image dimensions are unreasonable.")
    return image


def detect(data: bytes, content_type: str, confidence: float, iou: float) -> dict[str, Any]:
    image = decode_image(data, content_type)
    model, device, mode = get_model()
    names = _class_names(model)
    started = perf_counter()
    try:
        results = model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            max_det=settings.detector_max_detections,
            device=device,
            verbose=False,
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail="Detector inference failed.") from error
    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    width, height = image.size
    detections: list[dict[str, Any]] = []
    boxes = results[0].boxes if results else None
    if boxes is not None:
        for xyxy, class_value, confidence_value in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):
            source_index = int(class_value)
            source_name = names.get(source_index, f"class_{source_index}")
            direct_index = OBJECT_CLASSES.index(source_name) if source_name in OBJECT_CLASSES else None
            if direct_index is None and source_name not in GENERIC_CLASSES:
                continue
            suggested_index = direct_index
            if source_name == "sports ball":
                suggested_index = 7
            x1, y1, x2, y2 = xyxy
            x1, y1 = max(0.0, min(width, x1)), max(0.0, min(height, y1))
            x2, y2 = max(0.0, min(width, x2)), max(0.0, min(height, y2))
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            detections.append({
                "id": uuid4().hex,
                "source_class_index": source_index,
                "source_class_name": source_name,
                "confidence": round(float(confidence_value), 5),
                "x": x1 / width,
                "y": y1 / height,
                "width": box_width / width,
                "height": box_height / height,
                "suggested_class_index": suggested_index,
                "suggested_class_name": (
                    OBJECT_CLASSES[suggested_index] if suggested_index is not None else None
                ),
                "needs_classification": suggested_index is None,
                "source": "yolo",
                "model": _safe_model_name(),
            })
    detections.sort(key=lambda item: (-item["confidence"], item["id"]))
    return {
        "frame_width": width,
        "frame_height": height,
        "model": _safe_model_name(),
        "model_mode": mode,
        "device": device,
        "inference_time_ms": elapsed_ms,
        "detection_count": len(detections),
        "detections": detections,
    }
