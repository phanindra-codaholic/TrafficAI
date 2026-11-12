"""YOLOv8 vehicle detection."""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Any


CLASS_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


class AIDetector:
    """Runs YOLOv8 inference on camera images."""

    def __init__(self):
        self.session = None
        self.input_name = ""
        self.output_name = ""
        self.is_model_loaded = False

    def load_model(self, model_path: str) -> bool:
        """Load the ONNX model."""
        try:
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.is_model_loaded = True
            print("[AI] Model loaded successfully!")
            return True
        except Exception as e:
            print(f"[AI Error] Failed to load model: {e}")
            self.is_model_loaded = False
            return False

    def is_ready(self) -> bool:
        return self.is_model_loaded

    def _detect_on_patch(
        self,
        patch: np.ndarray,
        conf_threshold: float,
        offset_x: int,
        offset_y: int,
    ) -> Tuple[List[List[int]], List[float], List[int]]:
        """Detect vehicles in an image patch."""
        if patch.size == 0:
            return [], [], []

        blob = cv2.dnn.blobFromImage(
            patch, 1.0 / 255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False
        )

        outputs = self.session.run([self.output_name], {self.input_name: blob})
        det_mat = outputs[0][0]
        transposed = det_mat.transpose()

        boxes, confidences, class_ids = [], [], []
        h_orig, w_orig = patch.shape[:2]
        x_scale = w_orig / 640.0
        y_scale = h_orig / 640.0

        for row in transposed:
            x_center, y_center, w, h = row[:4]
            scores = row[4:84]

            max_class_id = int(np.argmax(scores))
            max_conf = float(scores[max_class_id])

            if max_conf > conf_threshold and max_class_id in CLASS_NAMES:
                left   = int((x_center - w / 2.0) * x_scale) + offset_x
                top    = int((y_center - h / 2.0) * y_scale) + offset_y
                width  = int(w * x_scale)
                height = int(h * y_scale)

                boxes.append([left, top, width, height])
                confidences.append(max_conf)
                class_ids.append(max_class_id)

        return boxes, confidences, class_ids

    def detect_vehicles(
        self, frame: np.ndarray, conf_threshold: float = 0.15
    ) -> List[Dict[str, Any]]:
        """Detect vehicles in a frame."""
        if not self.is_model_loaded or frame is None or frame.size == 0:
            return []

        all_boxes: List[List[int]] = []
        all_confs: List[float] = []
        all_ids:   List[int] = []

        b, c, ids = self._detect_on_patch(frame, conf_threshold, 0, 0)
        all_boxes.extend(b)
        all_confs.extend(c)
        all_ids.extend(ids)

        h_frame, w_frame = frame.shape[:2]
        tile_w = w_frame // 3
        tile_h = h_frame // 3
        overlap_x = tile_w // 4
        overlap_y = tile_h // 4

        y = 0
        while y <= h_frame - tile_h:
            x = 0
            while x <= w_frame - tile_w:
                if tile_w >= 100 and tile_h >= 100:
                    tile = frame[y : y + tile_h, x : x + tile_w]
                    b, c, ids = self._detect_on_patch(
                        tile, conf_threshold, x, y
                    )
                    all_boxes.extend(b)
                    all_confs.extend(c)
                    all_ids.extend(ids)
                x += tile_w - overlap_x
            y += tile_h - overlap_y

        if not all_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(all_boxes, all_confs, conf_threshold, 0.30)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    "class_id":   all_ids[i],
                    "class_name": CLASS_NAMES[all_ids[i]],
                    "confidence": all_confs[i],
                    "bbox":       all_boxes[i],
                })

        return detections
