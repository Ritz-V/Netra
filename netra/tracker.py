from __future__ import annotations

from typing import Dict, List

from .models import BBox, Detection, Track, current_time


def iou(boxA: BBox, boxB: BBox) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


class CentroidTracker:
    """Very simple IoU-based tracker.

    Good enough for a POC, and easy to explain.
    """

    def __init__(self, max_missing_time: float = 1.0, iou_threshold: float = 0.3):
        self.max_missing_time = max_missing_time
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        now = current_time()

        unmatched_dets = list(range(len(detections)))
        used_tracks: Dict[int, bool] = {tid: False for tid in self.tracks.keys()}

        # Match detections to existing tracks by IoU
        for det_idx, det in enumerate(detections):
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.tracks.items():
                if used_tracks[track_id]:
                    continue
                if track.class_name != det.class_name:
                    continue
                score = iou(track.bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                self.tracks[best_track_id].update(det.bbox, timestamp=now)
                used_tracks[best_track_id] = True
                if det_idx in unmatched_dets:
                    unmatched_dets.remove(det_idx)

        # New tracks
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                class_name=det.class_name,
                bbox=det.bbox,
                last_update_time=now,
            )

        # Remove stale tracks
        to_delete = [
            tid
            for tid, track in self.tracks.items()
            if now - track.last_update_time > self.max_missing_time
        ]
        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks