"""Shared helpers for smoothing, geometry, and timing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


Point2D = Tuple[float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if abs(in_max - in_min) < 1e-9:
        return out_min
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + (ratio * (out_max - out_min))


def distance_2d(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mean_point(points: Iterable[Point2D]) -> Point2D:
    points = list(points)
    if not points:
        return 0.0, 0.0
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    return sx / len(points), sy / len(points)


@dataclass
class ExponentialSmoother:
    """Simple exponential moving average smoother for cursor coordinates."""

    alpha: float
    x: Optional[float] = None
    y: Optional[float] = None

    def update(self, target_x: float, target_y: float, alpha: Optional[float] = None) -> Point2D:
        current_alpha = self.alpha if alpha is None else alpha
        if self.x is None or self.y is None:
            self.x = target_x
            self.y = target_y
        else:
            self.x = (1.0 - current_alpha) * self.x + current_alpha * target_x
            self.y = (1.0 - current_alpha) * self.y + current_alpha * target_y
        return self.x, self.y

    def reset(self) -> None:
        self.x = None
        self.y = None


class CooldownTimer:
    """Keeps action events from firing too frequently."""

    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger = 0.0

    def ready(self) -> bool:
        return (time.perf_counter() - self.last_trigger) >= self.cooldown_seconds

    def trigger(self) -> None:
        self.last_trigger = time.perf_counter()


class FPSCounter:
    """Lightweight FPS estimator with smoothing."""

    def __init__(self) -> None:
        self.prev = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        dt = now - self.prev
        self.prev = now
        instant = 1.0 / dt if dt > 1e-9 else 0.0
        self.fps = instant if self.fps == 0 else (0.90 * self.fps + 0.10 * instant)
        return self.fps


def finger_is_extended(landmarks: list[tuple[float, float, float]], tip_id: int, pip_id: int) -> bool:
    # MediaPipe y-axis increases downward. Tip above PIP means extended finger.
    return landmarks[tip_id][1] < landmarks[pip_id][1]
