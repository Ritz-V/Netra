from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


class EventType(str, Enum):
    VIOLENCE = "violence_aggression"
    CROWD_PANIC = "crowd_panic"
    FALL = "fall_collapse"
    CHILD_DISTRESS = "child_distress"
    ABANDONED_OBJECT = "abandoned_object"


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox: BBox
    last_update_time: float
    history: List[BBox] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)

    def update(self, bbox: BBox, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        # Store history
        self.history.append(self.bbox)
        # Compute velocity as centroid displacement
        cx_old = (self.bbox[0] + self.bbox[2]) / 2
        cy_old = (self.bbox[1] + self.bbox[3]) / 2
        cx_new = (bbox[0] + bbox[2]) / 2
        cy_new = (bbox[1] + bbox[3]) / 2
        dt = max(timestamp - self.last_update_time, 1e-3)
        vx = (cx_new - cx_old) / dt
        vy = (cy_new - cy_old) / dt
        self.velocities.append((vx, vy))
        # Update state
        self.bbox = bbox
        self.last_update_time = timestamp

    @property
    def speed(self) -> float:
        if not self.velocities:
            return 0.0
        vx, vy = self.velocities[-1]
        return (vx**2 + vy**2) ** 0.5

    @property
    def direction(self) -> Tuple[float, float]:
        if not self.velocities:
            return (0.0, 0.0)
        vx, vy = self.velocities[-1]
        mag = (vx**2 + vy**2) ** 0.5 or 1e-6
        return (vx / mag, vy / mag)

    @property
    def aspect_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox
        w = max(x2 - x1, 1e-3)
        h = max(y2 - y1, 1e-3)
        return h / w


@dataclass
class Event:
    timestamp: float
    frame_index: int
    event_type: EventType
    severity: int
    description: str
    track_ids: List[int] = field(default_factory=list)
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class FrameStatus:
    frame_index: int
    timestamp: float
    overall_severity: int
    color: str
    active_events: List[Event] = field(default_factory=list)


def current_time() -> float:
    """Wrapper for easier mocking in unit tests."""
    return time.time()