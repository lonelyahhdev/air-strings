"""Gesture-layer vocabulary: what the hands *mean*, independent of any instrument."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Hit(Enum):
    """What a pinch asks the instrument to play."""

    CHORD = "chord"  # thumb + index: the whole chord
    BASS = "bass"  # thumb + middle: a bass note
    ARP = "arp"  # thumb + ring: the chord as a rolling arpeggio


@dataclass(frozen=True)
class Trigger:
    """A pinch just closed."""

    t: float
    kind: Hit
    intensity: float  # 0..1 from how fast the fingers closed
    pos: tuple[float, float]  # where the pinch happened, normalised image coords

    @property
    def velocity(self) -> int:
        """MIDI velocity 24..127."""
        return int(round(24 + 103 * self.intensity))


@dataclass(frozen=True)
class Release:
    """The pinch that produced a :class:`Trigger` just opened."""

    t: float
    kind: Hit
    held_s: float


@dataclass(frozen=True)
class ChordIntent:
    """Current harmonic choice from the chord hand."""

    fingers: int  # committed (debounced) finger count
    degree: int | None  # scale degree 1..7; None = fist/unmapped = silence
    extended: bool  # raised hand -> add a 7th
    present: bool  # is the chord hand visible right now?
    changed: bool  # did degree/extension change on this frame?
