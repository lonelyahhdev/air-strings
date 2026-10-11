"""Chord hand: finger count picks the chord, hand height adds colour."""

from __future__ import annotations

from airstrings.config import GestureConfig
from airstrings.gestures.types import ChordIntent
from airstrings.tracking.features import count_fingers, palm_center
from airstrings.tracking.types import Hand


class ChordSelector:
    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self._committed: int | None = None
        self._candidate: int | None = None
        self._since = 0.0
        self._extended = False

    def _degree(self, fingers: int | None) -> int | None:
        return self.cfg.chord_map.get(fingers) if fingers is not None else None

    def update(self, hand: Hand | None, t: float) -> ChordIntent:
        if hand is None:  # keep the last chord ringing/selected when the hand leaves
            return ChordIntent(self._committed or 0, self._degree(self._committed), self._extended, False, False)

        before = (self._committed, self._extended)
        raw = count_fingers(hand)
        if raw != self._candidate:
            self._candidate, self._since = raw, t
        if raw != self._committed and t - self._since >= self.cfg.settle_s:
            self._committed = raw

        y = palm_center(hand)[1]
        if self._extended:
            if y > self.cfg.seventh_zone_y + self.cfg.seventh_zone_hysteresis:
                self._extended = False
        elif y < self.cfg.seventh_zone_y:
            self._extended = True

        changed = (self._committed, self._extended) != before
        return ChordIntent(self._committed or 0, self._degree(self._committed), self._extended, True, changed)
