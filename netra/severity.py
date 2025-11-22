from __future__ import annotations

from typing import Dict


def scale_to_severity(value: float, low: float, high: float, max_severity: int = 10) -> int:
    """Map a continuous value to a 0–max_severity integer.

    Values below `low` → 0, above `high` → max_severity.
    """
    if value <= low:
        return 0
    if value >= high:
        return max_severity
    frac = (value - low) / max(1e-6, (high - low))
    return int(frac * max_severity)


def combine_signals(signals: Dict[str, float]) -> int:
    """Very simple severity combiner – just takes the max."""
    if not signals:
        return 0
    return int(max(signals.values()))