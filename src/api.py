"""TrafficAI REST API."""

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

from control.traffic_manager import TrafficManager

app = FastAPI(title="TrafficAI", version="1.0.0")
manager = TrafficManager(model_path="../models/yolov8n_traffic.onnx")


class SignalResult(BaseModel):
    road_id: int
    vehicle_count: int
    green_time: int
    priority_rank: int
    is_emergency: bool


class PipelineResponse(BaseModel):
    total_cameras: int
    total_vehicles: int
    cycle_time: int
    signal_plan: List[SignalResult]


@app.get("/health")
def health():
    """Return model status."""
    return {"status": "ok", "model_loaded": manager.detector.is_ready()}


@app.post("/detect", response_model=PipelineResponse)
async def detect(cameras: List[UploadFile] = File(...)):
    """Process one to four camera images."""
    if not manager.detector.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(cameras) < 1 or len(cameras) > 4:
        raise HTTPException(status_code=400, detail="Send 1 to 4 camera images")

    frames = []
    for cam in cameras:
        data = await cam.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Invalid image: {cam.filename}")
        frames.append(img)

    cycle_plan = manager.process_intersection(frames)

    signal_plan = [
        SignalResult(
            road_id=dec.lane_id,
            vehicle_count=dec.vehicle_count,
            green_time=dec.green_time,
            priority_rank=dec.priority_rank,
            is_emergency=dec.is_emergency,
        )
        for dec in cycle_plan
    ]

    return PipelineResponse(
        total_cameras=len(frames),
        total_vehicles=sum(s.vehicle_count for s in signal_plan),
        cycle_time=sum(s.green_time for s in signal_plan),
        signal_plan=signal_plan,
    )
