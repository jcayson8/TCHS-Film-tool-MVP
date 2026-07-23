"""OpenCV tracking routes for accepted coach annotations."""

import json
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas import TrackingResponse
from app.services import tracker

router = APIRouter(tags=["tracking"])


@router.post("/track/frames", response_model=TrackingResponse)
async def track_frames(
    initial_image: Annotated[UploadFile, File()],
    frames: Annotated[list[UploadFile], File()],
    boxes: Annotated[str, Form()],
    frame_times: Annotated[str, Form()],
) -> dict:
    try:
        times = json.loads(frame_times)
    except (TypeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail="frame_times must be valid JSON.") from error
    if not isinstance(times, list):
        raise HTTPException(status_code=400, detail="frame_times must be a JSON array.")
    initial_data = await initial_image.read(12 * 1024 * 1024 + 1)
    payloads = [
        (await frame.read(12 * 1024 * 1024 + 1), frame.content_type or "")
        for frame in frames
    ]
    return await run_in_threadpool(
        tracker.track,
        initial_data,
        initial_image.content_type or "",
        payloads,
        boxes,
        times,
    )
