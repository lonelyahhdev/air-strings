"""Rhythm hand: a palm swing through the strum line fires a trigger."""

from __future__ import annotations

from airstrings.config import GestureConfig
from airstrings.gestures.types import Direction, Trigger
from airstrings.tracking.features import hand_scale, openness, palm_center
from airstrings.tracking.smoothing import OneEuroFilter
from airstrings.tracking.types import Hand


class StrumDetector:
    """Detects palm crossings of a horizontal line.

    Speed is measured in hand-lengths/second, so dynamics don't depend on how
    far the player sits from the camera.  Slow drift across the line is
    ignored (``min_strum_speed``), and a refractory period swallows jitter.
    """

    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self.line_y = cfg.strum_line_y
        self._filter = OneEuroFilter(cfg.smoothing_min_cutoff, cfg.smoothing_beta)
        self._prev: tuple[float, float] | None = None  # (t, filtered y)
        self._last_fire = -1e9

    def reset(self) -> None:
        self._filter.reset()
        self._prev = None

    def update(self, hand: Hand | None, t: float) -> list[Trigger]:
        if hand is None:
            self.reset()
            return []
        if self._prev is not None and t - self._prev[0] > self.cfg.lost_gap_s:
            self.reset()

        x, raw_y = palm_center(hand)
        y = self._filter(raw_y, t)
        prev, self._prev = self._prev, (t, y)
        if prev is None or t <= prev[0]:
            return []

        pt, py = prev
        line = self.line_y
        if py < line <= y:
            direction = Direction.DOWN
        elif py > line >= y:
            direction = Direction.UP
        else:
            return []

        speed = abs(y - py) / (t - pt) / hand_scale(hand)
        if speed < self.cfg.min_strum_speed or t - self._last_fire < self.cfg.refractory_s:
            return []

        span = max(self.cfg.full_strum_speed - self.cfg.min_strum_speed, 1e-6)
        intensity = min(max((speed - self.cfg.min_strum_speed) / span, 0.0), 1.0) ** 0.7
        self._last_fire = t
        return [Trigger(t, direction, intensity, x, openness(hand))]
