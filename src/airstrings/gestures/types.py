"""Gesture-layer vocabulary: what the hands *mean*, independent of any instrument."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    DOWN = "down"
    UP = "up"


@dataclass(frozen=True)
class Trigger:
    """A rhythmic hit from the playing hand."""

    t: float
    direction: Direction
    intensity: float  # 0..1 from swing speed
    x: float  # horizontal hand position, 0..1 in the mirrored view
    openness: float  # 0 = closed fist (damped) .. 1 = open hand (ringing)

    @property
    def velocity(self) -> int:
        """MIDI velocity 24..127."""
        return int(round(24 + 103 * self.intensity))


@dataclass(frozen=True)
class ChordIntent:
    """Current harmonic choice from the chord hand."""

    fingers: int  # committed (debounced) finger count
    degree: int | None  # scale degree 1..7; None = fist/unmapped = silence
    extended: bool  # raised hand -> add a 7th
    present: bool  # is the chord hand visible right now?
    changed: bool  # did degree/extension change on this frame?
