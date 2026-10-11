"""Geometric hand features, independent of camera distance and rotation."""

from __future__ import annotations

import numpy as np

from airstrings.tracking.types import LM, Hand

_FINGERS = (
    (LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_TIP),
    (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP),
    (LM.RING_MCP, LM.RING_PIP, LM.RING_TIP),
    (LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_TIP),
)
_PALM_POINTS = [LM.WRIST, LM.INDEX_MCP, LM.MIDDLE_MCP, LM.RING_MCP, LM.PINKY_MCP]

FINGER_EXTENDED_RATIO = 1.05  # tip is this much farther from wrist than the PIP
THUMB_EXTENDED_RATIO = 1.05


def iso_xy(hand: Hand) -> np.ndarray:
    """(21, 2) landmark coords in image-height units (x scaled by aspect)."""
    return hand.landmarks[:, :2] * np.array([hand.aspect, 1.0], dtype=np.float32)


def _dist(pts: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(pts[a] - pts[b]))


def hand_scale(hand: Hand) -> float:
    """Wrist -> middle-finger knuckle, in image-height units (>0)."""
    return max(_dist(iso_xy(hand), LM.WRIST, LM.MIDDLE_MCP), 1e-4)


def palm_center(hand: Hand) -> tuple[float, float]:
    """Mean of the palm landmarks in normalised image coords. Stable vs finger motion."""
    x, y = hand.landmarks[_PALM_POINTS, :2].mean(axis=0)
    return float(x), float(y)


def finger_states(hand: Hand) -> tuple[bool, bool, bool, bool, bool]:
    """(thumb, index, middle, ring, pinky) extended?"""
    pts = iso_xy(hand)
    states = []
    thumb_out = _dist(pts, LM.THUMB_TIP, LM.PINKY_MCP)
    thumb_in = _dist(pts, LM.THUMB_IP, LM.PINKY_MCP)
    states.append(thumb_out > thumb_in * THUMB_EXTENDED_RATIO)
    for _mcp, pip, tip in _FINGERS:
        states.append(_dist(pts, LM.WRIST, tip) > _dist(pts, LM.WRIST, pip) * FINGER_EXTENDED_RATIO)
    return tuple(states)  # type: ignore[return-value]


def count_fingers(hand: Hand) -> int:
    return sum(finger_states(hand))


def is_fist(hand: Hand) -> bool:
    return count_fingers(hand) == 0


def pinch_ratios(hand: Hand) -> tuple[float, float, float, float]:
    """Thumb-tip to fingertip distance for (index, middle, ring, pinky), in hand-lengths.

    ~1.0 for a relaxed open hand, below ~0.3 when the fingertips touch.
    """
    pts = iso_xy(hand)
    scale = hand_scale(hand)
    return tuple(_dist(pts, LM.THUMB_TIP, tip) / scale for _mcp, _pip, tip in _FINGERS)  # type: ignore[return-value]


def finger_reach(hand: Hand, tip: int) -> float:
    """Wrist-to-fingertip distance in hand-lengths. Small when a finger is curled into a fist."""
    return _dist(iso_xy(hand), LM.WRIST, tip) / hand_scale(hand)
