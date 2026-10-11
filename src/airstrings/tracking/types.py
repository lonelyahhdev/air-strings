"""Tracker-agnostic hand data. Everything downstream depends only on this."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"

    @property
    def other(self) -> Side:
        return Side.RIGHT if self is Side.LEFT else Side.LEFT


class LM(IntEnum):
    """The 21 MediaPipe hand landmark indices."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass(frozen=True)
class Hand:
    """One tracked hand.

    ``landmarks`` is (21, 3): x, y normalised to the *displayed (mirrored)*
    image in [0, 1], y growing downward; z is relative depth.  ``aspect`` is
    image width / height so distances can be made isotropic.
    """

    side: Side
    landmarks: np.ndarray
    score: float = 1.0
    aspect: float = 16 / 9


@dataclass(frozen=True)
class TrackingFrame:
    t: float  # seconds, monotonic
    hands: tuple[Hand, ...] = field(default_factory=tuple)

    def hand(self, side: Side) -> Hand | None:
        for hand in self.hands:
            if hand.side is side:
                return hand
        return None
