from __future__ import annotations

from typing import Iterable, List, Tuple
import math

from .models import Event, EventType, FrameStatus, Track, current_time
from .severity import combine_signals, scale_to_severity


def _distance(a: Track, b: Track) -> float:
    xa1, ya1, xa2, ya2 = a.bbox
    xb1, yb1, xb2, yb2 = b.bbox
    cax, cay = (xa1 + xa2) / 2, (ya1 + ya2) / 2
    cbx, cby = (xb1 + xb2) / 2, (yb1 + yb2) / 2
    return math.hypot(cax - cbx, cay - cby)


def detect_violence_aggression(frame_index: int, person_tracks: Iterable[Track]) -> List[Event]:
    """Rough heuristic for aggressive interactions."""
    tracks = [t for t in person_tracks]
    events: List[Event] = []
    n = len(tracks)
    now = current_time()

    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracks[i], tracks[j]
            dist = _distance(a, b)
            if dist <= 0:
                continue
            rel_speed = a.speed + b.speed
            da = a.direction
            db = b.direction
            dot = da[0] * db[0] + da[1] * db[1]

            if dist < 80 and rel_speed > 100 and dot < 0:
                sev_speed = scale_to_severity(rel_speed, low=80, high=250)
                sev_dist = scale_to_severity(80 - dist, low=0, high=80)
                severity = combine_signals({"speed": sev_speed, "proximity": sev_dist})
                if severity > 0:
                    events.append(
                        Event(
                            timestamp=now,
                            frame_index=frame_index,
                            event_type=EventType.VIOLENCE,
                            severity=min(10, max(2, severity)),
                            description=(
                                "Rapid close-proximity movement between people "
                                "(possible aggression)."
                            ),
                            track_ids=[a.track_id, b.track_id],
                            extra={"distance_px": float(dist), "rel_speed": float(rel_speed)},
                        )
                    )
    return events


def detect_crowd_panic(frame_index: int, person_tracks: Iterable[Track]) -> List[Event]:
    tracks = [t for t in person_tracks]
    events: List[Event] = []
    now = current_time()

    if len(tracks) < 3:
        return events

    speeds = [t.speed for t in tracks]
    avg_speed = sum(speeds) / max(len(speeds), 1)

    dirs = [t.direction for t in tracks]
    if dirs:
        avg_dx = sum(d[0] for d in dirs) / len(dirs)
        avg_dy = sum(d[1] for d in dirs) / len(dirs)
        dir_mag = math.hypot(avg_dx, avg_dy)
    else:
        dir_mag = 0.0

    sev_speed = scale_to_severity(avg_speed, low=40, high=200)
    sev_count = scale_to_severity(len(tracks), low=3, high=15)
    sev_coherence = scale_to_severity(dir_mag, low=0.2, high=1.0)

    severity = combine_signals(
        {"speed": sev_speed, "crowd_size": sev_count, "direction": sev_coherence}
    )

    if severity <= 0:
        return events

    events.append(
        Event(
            timestamp=now,
            frame_index=frame_index,
            event_type=EventType.CROWD_PANIC,
            severity=severity,
            description="Unusually fast, dense crowd movement (possible panic).",
            track_ids=[t.track_id for t in tracks],
            extra={
                "avg_speed": float(avg_speed),
                "crowd_size": float(len(tracks)),
                "direction_coherence": float(dir_mag),
            },
        )
    )

    return events


def detect_falls(frame_index: int, person_tracks: Iterable[Track]) -> List[Event]:
    events: List[Event] = []
    now = current_time()
    for t in person_tracks:
        aspect = t.aspect_ratio
        if len(t.history) < 2:
            continue
        prev_box = t.history[-1]
        x1, y1, x2, y2 = t.bbox
        px1, py1, px2, py2 = prev_box
        dy = (y2 - y1) - (py2 - py1)
        speed = t.speed

        became_horizontal = aspect < 1.0
        strong_downward = dy > 20 and speed > 60

        if became_horizontal and strong_downward:
            sev_motion = scale_to_severity(dy, low=20, high=120)
            severity = max(2, sev_motion)
            events.append(
                Event(
                    timestamp=now,
                    frame_index=frame_index,
                    event_type=EventType.FALL,
                    severity=min(10, severity),
                    description=(
                        "Rapid vertical movement followed by horizontal posture "
                        "(possible fall / collapse)."
                    ),
                    track_ids=[t.track_id],
                    extra={"aspect_ratio": float(aspect), "dy": float(dy)},
                )
            )
    return events


def detect_child_distress(frame_index: int, person_tracks: Iterable[Track]) -> List[Event]:
    tracks = [t for t in person_tracks]
    events: List[Event] = []
    now = current_time()

    if len(tracks) < 2:
        return events

    def height(t: Track) -> float:
        return t.bbox[3] - t.bbox[1]

    for child in tracks:
        for adult in tracks:
            if child.track_id == adult.track_id:
                continue
            if height(child) >= 0.75 * height(adult):
                continue

            dist = _distance(child, adult)
            if dist > 80:
                continue

            ch_dir = child.direction
            ad_dir = adult.direction
            dot = ch_dir[0] * ad_dir[0] + ch_dir[1] * ad_dir[1]
            ad_speed = adult.speed
            ch_speed = child.speed

            if dot > 0.7 and ad_speed > 80 and ch_speed > 40:
                sev_speed = scale_to_severity(ad_speed, low=80, high=200)
                sev_prox = scale_to_severity(80 - dist, low=0, high=80)
                severity = combine_signals({"speed": sev_speed, "proximity": sev_prox})
                if severity > 0:
                    events.append(
                        Event(
                            timestamp=now,
                            frame_index=frame_index,
                            event_type=EventType.CHILD_DISTRESS,
                            severity=min(10, max(3, severity)),
                            description=(
                                "Small person moving rapidly alongside larger person "
                                "(possible pulling/dragging)."
                            ),
                            track_ids=[child.track_id, adult.track_id],
                            extra={
                                "distance_px": float(dist),
                                "adult_speed": float(ad_speed),
                                "child_speed": float(ch_speed),
                            },
                        )
                    )

    return events


def detect_abandoned_objects(
    frame_index: int,
    bag_tracks: Iterable[Track],
    person_tracks: Iterable[Track],
    abandoned_after_s: float = 10.0,
    alone_radius_px: float = 120.0,
) -> List[Event]:
    """Flag bags that are static & have no nearby person for a while."""
    now = current_time()
    events: List[Event] = []
    persons = list(person_tracks)

    def centroid(t: Track) -> Tuple[float, float]:
        x1, y1, x2, y2 = t.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    for bag in bag_tracks:
        speed = bag.speed
        if speed > 10:
            continue

        age_s = max(0.0, now - bag.last_update_time)
        if age_s < abandoned_after_s:
            continue

        bx, by = centroid(bag)
        min_dist = float("inf")
        for p in persons:
            px, py = centroid(p)
            d = math.hypot(bx - px, by - py)
            min_dist = min(min_dist, d)

        if min_dist < alone_radius_px:
            continue

        sev_time = scale_to_severity(age_s, low=abandoned_after_s, high=3 * abandoned_after_s)
        severity = max(3, sev_time)

        events.append(
            Event(
                timestamp=now,
                frame_index=frame_index,
                event_type=EventType.ABANDONED_OBJECT,
                severity=min(10, severity),
                description=(
                    "Static bag/item left alone with no nearby person "
                    "(possible abandoned object)."
                ),
                track_ids=[bag.track_id],
                extra={"age_s": float(age_s), "min_person_dist": float(min_dist)},
            )
        )

    return events


def aggregate_frame_status(frame_index: int, events: List[Event]) -> FrameStatus:
    """Compute overall severity + colour code for a frame from its events."""
    if events:
        overall = max(e.severity for e in events)
    else:
        overall = 0

    if overall <= 2:
        color = "green"
    elif overall <= 5:
        color = "yellow"
    else:
        color = "red"

    return FrameStatus(
        frame_index=frame_index,
        timestamp=current_time(),
        overall_severity=overall,
        color=color,
        active_events=events,
    )
