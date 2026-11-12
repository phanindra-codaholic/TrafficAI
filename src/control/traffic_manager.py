"""Traffic detection and signal planning."""

import cv2
import numpy as np
from typing import List

from ai.detector import AIDetector
from control.optimization_engine import OptimizationEngine, LaneData, SignalDecision


class TrafficManager:
    """Coordinate detection and signal scheduling."""

    def __init__(self, model_path: str = "../models/yolov8n_traffic.onnx"):
        self.detector = AIDetector()
        self.engine = OptimizationEngine()
        self.detector.load_model(model_path)

    def process_intersection(
        self, camera_feeds: List[np.ndarray]
    ) -> List[SignalDecision]:
        """Create a signal plan for the supplied camera images."""
        lane_id_start = 101
        current_lane_data: List[LaneData] = []

        for i, frame in enumerate(camera_feeds):
            data = LaneData(
                lane_id=lane_id_start + i,
                vehicle_count=0,
                wait_time=10 + (i * 5),
                has_emergency=False,
            )

            if self.detector.is_ready() and frame is not None and frame.size > 0:
                detections = self.detector.detect_vehicles(frame, conf_threshold=0.25)

                img_h, img_w = frame.shape[:2]
                roi_top    = 0
                roi_bottom = int(img_h * 0.98)
                roi_left   = int(img_w * 0.20)
                roi_right  = int(img_w * 0.80)
                road_area  = (roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top)

                cv2.rectangle(
                    frame,
                    (road_area[0], road_area[1]),
                    (road_area[0] + road_area[2], road_area[1] + road_area[3]),
                    (255, 0, 0), 1,
                )

                for det in detections:
                    bx, by, bw, bh = det["bbox"]
                    cx = bx + bw // 2
                    cy = by + bh // 2

                    inside = (
                        road_area[0] <= cx <= road_area[0] + road_area[2]
                        and road_area[1] <= cy <= road_area[1] + road_area[3]
                    )

                    if inside:
                        data.vehicle_count += 1
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)

                label = f"Road {data.lane_id}: {data.vehicle_count} vehicles"
                cv2.putText(
                    frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )

            current_lane_data.append(data)

        return self.engine.calculate_signal_cycle(current_lane_data)
