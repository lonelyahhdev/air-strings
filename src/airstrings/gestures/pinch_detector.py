"""Playing hand: touching the thumb to a fingertip plays something.

Only the fingers move, so the hand can rest.  Which fingertip decides *what*
plays; how fast the fingers close decides *how loud*; holding the pinch
sustains and letting go ends the hold.
"""

from __future__ import annotations

from collections import deque

from airstrings.config import GestureConfig
from airstrings.gestures.types import Hit, Release, Trigger
from airstrings.tracking.features import finger_reach, pinch_ratios
from airstrings.tracking.types import LM, Hand

# index, middle, ring, pinky -> what a pinch with that finger plays (pinky: reserved, ignored)
_KINDS: tuple[Hit | None, ...] = (Hit.CHORD, Hit.BASS, Hit.ARP, None)
_TIPS = (LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP)
_HISTORY_S = 0.25


class PinchDetector:
    """Hysteresis state machine over the thumb-to-fingertip distances.

    On a flat hand the fingertips sit close together, so touching the middle
    finger also brings the index and ring tips near the thumb.  A pinch only
    fires once one fingertip is clearly the nearest (by ``pinch_margin``), and
    the pinky is considered too so a pinky pinch cannot be misread as ring.
    """

    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self.on = cfg.pinch_on
        self.off = cfg.pinch_off
        self._active: int | None = None
        self._since = 0.0
        self._last_fire = -1e9
        self._history: deque[tuple[float, tuple[float, ...]]] = deque()
        self.ratios: tuple[float, ...] | None = None  # latest thumb-to-tip distances, for the UI

    @property
    def active(self) -> Hit | None:
        return None if self._active is None else _KINDS[self._active]

    def nudge(self, delta: float) -> None:
        """Make pinching easier (+) or harder (-), keeping the open/close gap."""
        gap = self.off - self.on
        self.on = min(max(self.on + delta, 0.12), 0.6)
        self.off = self.on + gap

    def reset(self) -> None:
        self._active = None
        self._history.clear()
        self.ratios = None

    def _end(self, t: float) -> list[Release]:
        if self._active is None:
            return []
        kind = _KINDS[self._active]
        held = t - self._since
        self._active = None
        return [Release(t, kind, held)] if kind is not None else []

    def update(self, hand: Hand | None, t: float) -> list[Trigger | Release]:
        if hand is None:
            events = self._end(t)  # hand left the picture mid-pinch: let go
            self.reset()
            return events

        ratios = pinch_ratios(hand)
        self.ratios = ratios
        self._history.append((t, ratios))
        while self._history and t - self._history[0][0] > _HISTORY_S:
            self._history.popleft()

        if self._active is not None:
            if ratios[self._active] > self.off:
                return list(self._end(t))
            return []

        order = sorted(range(4), key=ratios.__getitem__)
        best, second = order[0], order[1]
        kind = _KINDS[best]
        if (
            kind is None
            or ratios[best] >= self.on
            or ratios[second] - ratios[best] < self.cfg.pinch_margin
            or finger_reach(hand, _TIPS[best]) < self.cfg.pinch_min_reach
            or t - self._last_fire < self.cfg.pinch_refractory_s
        ):
            return []

        self._active, self._since, self._last_fire = best, t, t
        thumb = hand.landmarks[LM.THUMB_TIP]
        return [Trigger(t, kind, self._intensity(best, t, ratios[best]), (float(thumb[0]), float(thumb[1])))]

    def _intensity(self, finger: int, t: float, ratio_now: float) -> float:
        """Closing speed (hand-lengths/second) over the last quarter second -> 0..1."""
        t_max, r_max = max(((tt, r[finger]) for tt, r in self._history), key=lambda x: x[1])
        speed = max(r_max - ratio_now, 0.0) / max(t - t_max, 1 / 30)
        cfg = self.cfg
        span = max(cfg.pinch_fast_speed - cfg.pinch_slow_speed, 1e-6)
        x = min(max((speed - cfg.pinch_slow_speed) / span, 0.0), 1.0)
        return cfg.min_intensity + (1.0 - cfg.min_intensity) * x
