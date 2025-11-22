from __future__ import annotations

from typing import List

import numpy as np
import torch

from .models import Detection


class YoloV5Detector:
    """Thin wrapper around YOLOv5s via torch hub.

    This uses generic COCO classes and **never** performs face recognition.
    """

    def __init__(self, device: str = "cpu", conf_threshold: float = 0.3) -> None:
        self.device = device
        self.conf_threshold = conf_threshold
        # Lazy-load model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = (
                torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
                .to(self.device)
                .eval()
            )
        return self._model

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run YOLOv5 on a BGR frame and return filtered detections.

        Only keeps people and common bag-like objects.
        """
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self.model(frame_rgb)
        detections: List[Detection] = []

        for *xyxy, conf, cls in results.xyxy[0].tolist():  # type: ignore[attr-defined]
            confidence = float(conf)
            if confidence < self.conf_threshold:
                continue
            class_id = int(cls)
            class_name = self.model.names[class_id]  # type: ignore[index]

            allowed = {"person", "backpack", "handbag", "suitcase"}
            if class_name not in allowed:
                continue

            x1, y1, x2, y2 = map(float, xyxy)
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return detections