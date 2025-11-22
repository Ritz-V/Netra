from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from .behaviors import (
    aggregate_frame_status,
    detect_abandoned_objects,
    detect_child_distress,
    detect_crowd_panic,
    detect_falls,
    detect_violence_aggression,
)
from .models import Event, FrameStatus, Track
from .tracker import CentroidTracker
from .yolo_detector import YoloV5Detector


def draw_overlays(frame: np.ndarray, status: FrameStatus, tracks: List[Track]) -> np.ndarray:
    """Draw bounding boxes, track IDs, and status banner on the frame."""
    out = frame.copy()

    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        color = (0, 255, 0) if t.class_name == "person" else (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{t.class_name}#{t.track_id}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    banner_color = {
        "green": (0, 180, 0),
        "yellow": (0, 200, 200),
        "red": (0, 0, 255),
    }.get(status.color, (50, 50, 50))

    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 40), banner_color, -1)
    text = (
        f"Severity: {status.overall_severity} | Status: {status.color.upper()} | "
        f"Events: {len(status.active_events)}"
    )
    cv2.putText(
        out,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def run_stream(
    source: int | str,
    enable_modules: Optional[dict] = None,
) -> Generator[Tuple[int, np.ndarray, List[Event], FrameStatus], None, None]:
    """Main processing loop.

    Yields (frame_index, annotated_frame_bgr, events, status).
    """
    if enable_modules is None:
        enable_modules = {
            "violence": True,
            "crowd_panic": True,
            "fall": True,
            "child_distress": True,
            "abandoned": True,
        }

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    detector = YoloV5Detector(device="cpu")
    tracker = CentroidTracker()

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            tracks_dict = tracker.update(detections)
            person_tracks = [t for t in tracks_dict.values() if t.class_name == "person"]
            bag_tracks = [
                t
                for t in tracks_dict.values()
                if t.class_name in {"backpack", "handbag", "suitcase"}
            ]

            events: List[Event] = []

            if enable_modules.get("violence"):
                events.extend(detect_violence_aggression(frame_index, person_tracks))

            if enable_modules.get("crowd_panic"):
                events.extend(detect_crowd_panic(frame_index, person_tracks))

            if enable_modules.get("fall"):
                events.extend(detect_falls(frame_index, person_tracks))

            if enable_modules.get("child_distress"):
                events.extend(detect_child_distress(frame_index, person_tracks))

            if enable_modules.get("abandoned"):
                events.extend(
                    detect_abandoned_objects(
                        frame_index, bag_tracks=bag_tracks, person_tracks=person_tracks
                    )
                )

            status = aggregate_frame_status(frame_index, events)
            annotated = draw_overlays(frame, status, list(tracks_dict.values()))

            yield frame_index, annotated, events, status

            frame_index += 1
    finally:
        cap.release()
