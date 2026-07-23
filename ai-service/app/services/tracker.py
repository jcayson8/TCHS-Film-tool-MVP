"""Classical OpenCV tracking for coach-accepted annotations."""

from __future__ import annotations

import json
import math
from typing import Any

from fastapi import HTTPException

MAX_TRACKED_FRAMES = 60
MAX_TRACKED_BOXES = 11


def _opencv() -> Any:
    try:
        import cv2
        import numpy as np
    except (ImportError, ModuleNotFoundError) as error:
        raise HTTPException(
            status_code=503,
            detail="OpenCV tracking is unavailable. Install requirements-vision.txt.",
        ) from error
    return cv2, np


def _tracker_factory(cv2: Any) -> tuple[str, Any]:
    namespaces = (cv2, getattr(cv2, "legacy", None))
    for name in ("CSRT", "KCF", "MOSSE"):
        constructor_name = f"Tracker{name}_create"
        for namespace in namespaces:
            constructor = getattr(namespace, constructor_name, None) if namespace else None
            if callable(constructor):
                return name, constructor
    raise HTTPException(
        status_code=503,
        detail="No supported OpenCV tracker is available (CSRT, KCF, or MOSSE).",
    )


def _decode_image(data: bytes, content_type: str, cv2: Any, np: Any) -> Any:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded tracking frame is empty.")
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded tracking frame exceeds 12 MB.")
    if content_type.lower() not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Use JPEG, PNG, or WebP tracking frames.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise HTTPException(status_code=400, detail="A tracking frame could not be decoded.")
    return image


def _parse_boxes(raw_boxes: str) -> list[dict[str, Any]]:
    try:
        boxes = json.loads(raw_boxes)
    except (TypeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail="boxes must be valid JSON.") from error
    if not isinstance(boxes, list) or not 1 <= len(boxes) <= MAX_TRACKED_BOXES:
        raise HTTPException(
            status_code=400,
            detail=f"Track between 1 and {MAX_TRACKED_BOXES} accepted annotations.",
        )
    player_ids: set[str] = set()
    for box in boxes:
        if not isinstance(box, dict):
            raise HTTPException(status_code=400, detail="Each tracking box must be an object.")
        player_id = str(box.get("player_id", "")).strip()
        class_index = box.get("class_index")
        values = [box.get(key) for key in ("x", "y", "width", "height")]
        if (
            not player_id
            or player_id in player_ids
            or not isinstance(class_index, int)
            or not 0 <= class_index <= 8
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
        ):
            raise HTTPException(status_code=400, detail="Tracking boxes contain invalid values.")
        x, y, width, height = (float(value) for value in values)
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise HTTPException(status_code=400, detail="Tracking boxes must use valid normalized coordinates.")
        player_ids.add(player_id)
    return boxes


def _pixel_box(box: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    return (
        float(box["x"]) * width,
        float(box["y"]) * height,
        float(box["width"]) * width,
        float(box["height"]) * height,
    )


def _clamped_box(raw_box: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, box_width, box_height = (float(value) for value in raw_box)
    left = max(0, min(width - 1, round(x)))
    top = max(0, min(height - 1, round(y)))
    right = max(left + 1, min(width, round(x + box_width)))
    bottom = max(top + 1, min(height, round(y + box_height)))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right - left, bottom - top


def _patch(image: Any, box: tuple[int, int, int, int], cv2: Any) -> Any:
    x, y, width, height = box
    crop = image[y:y + height, x:x + width]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 96), interpolation=cv2.INTER_AREA)


def _confidence(previous: Any, current: Any, cv2: Any) -> tuple[str, float]:
    if previous is None or current is None:
        return "low", 0.0
    template_score = float(cv2.matchTemplate(previous, current, cv2.TM_CCOEFF_NORMED)[0][0])
    previous_hist = cv2.calcHist([previous], [0], None, [32], [0, 256])
    current_hist = cv2.calcHist([current], [0], None, [32], [0, 256])
    histogram_score = (float(cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL)) + 1) / 2
    score = max(template_score, histogram_score)
    if not math.isfinite(score):
        score = 0.0
    score = max(0.0, min(1.0, score))
    label = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    return label, round(score, 4)


def track(
    initial_data: bytes,
    initial_content_type: str,
    frame_payloads: list[tuple[bytes, str]],
    raw_boxes: str,
    frame_times: list[int],
) -> dict[str, Any]:
    cv2, np = _opencv()
    tracker_name, create_tracker = _tracker_factory(cv2)
    boxes = _parse_boxes(raw_boxes)
    if not 1 <= len(frame_payloads) <= MAX_TRACKED_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Provide between 1 and {MAX_TRACKED_FRAMES} tracking frames.",
        )
    if len(frame_times) != len(frame_payloads) or any(
        not isinstance(value, int) or value < 0 for value in frame_times
    ):
        raise HTTPException(status_code=400, detail="frame_times must match the uploaded frames.")

    initial = _decode_image(initial_data, initial_content_type, cv2, np)
    initial_height, initial_width = initial.shape[:2]
    active: dict[str, dict[str, Any]] = {}
    for box in boxes:
        pixel_box = _clamped_box(_pixel_box(box, initial_width, initial_height), initial_width, initial_height)
        if pixel_box is None:
            raise HTTPException(status_code=400, detail="A tracking box is too small.")
        instance = create_tracker()
        initialized = instance.init(initial, pixel_box)
        if initialized is False:
            raise HTTPException(status_code=422, detail=f"Could not initialize tracker for {box['player_id']}.")
        active[box["player_id"]] = {
            "tracker": instance,
            "class_index": box["class_index"],
            "patch": _patch(initial, pixel_box, cv2),
        }

    tracked_frames: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for frame_index, ((data, content_type), frame_time_ms) in enumerate(
        zip(frame_payloads, frame_times), start=1
    ):
        image = _decode_image(data, content_type, cv2, np)
        height, width = image.shape[:2]
        annotations: list[dict[str, Any]] = []
        for player_id in list(active):
            item = active[player_id]
            success, raw_box = item["tracker"].update(image)
            pixel_box = _clamped_box(raw_box, width, height) if success else None
            if pixel_box is None:
                failures.append({
                    "player_id": player_id,
                    "frame_index": frame_index,
                    "frame_time_ms": frame_time_ms,
                    "reason": "Tracker lost the player.",
                })
                del active[player_id]
                continue
            current_patch = _patch(image, pixel_box, cv2)
            confidence_label, confidence_score = _confidence(item["patch"], current_patch, cv2)
            item["patch"] = current_patch
            x, y, box_width, box_height = pixel_box
            annotations.append({
                "player_id": player_id,
                "class_index": item["class_index"],
                "x": x / width,
                "y": y / height,
                "width": box_width / width,
                "height": box_height / height,
                "tracking_confidence": confidence_label,
                "confidence_score": confidence_score,
            })
        tracked_frames.append({
            "frame_index": frame_index,
            "frame_time_ms": frame_time_ms,
            "annotations": annotations,
        })
        if not active:
            break
    return {
        "tracker": tracker_name,
        "requested_frame_count": len(frame_payloads),
        "completed_frame_count": len(tracked_frames),
        "frames": tracked_frames,
        "failures": failures,
    }
