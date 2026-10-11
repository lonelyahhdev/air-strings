"""The MediaPipe wrapper's conversion logic, tested with stand-ins (no model/GPU needed)."""

from types import SimpleNamespace

import numpy as np

from airstrings.tracking.mediapipe_tracker import MediaPipeHandTracker, _resolve_sides
from airstrings.tracking.synthetic import make_hand
from airstrings.tracking.types import Side


def fake_result(*hands):
    """hands: (label, x_offset)"""
    lms, handed = [], []
    for label, x in hands:
        lms.append([SimpleNamespace(x=x + 0.001 * i, y=0.5, z=0.0) for i in range(21)])
        handed.append([SimpleNamespace(category_name=label, score=0.9)])
    return SimpleNamespace(hand_landmarks=lms, handedness=handed)


def make_tracker(result):
    tracker = object.__new__(MediaPipeHandTracker)
    tracker._mp = SimpleNamespace(
        Image=lambda image_format, data: data, ImageFormat=SimpleNamespace(SRGB=1)
    )
    tracker._landmarker = SimpleNamespace(detect_for_video=lambda image, ts: result, close=lambda: None)
    tracker._last_ms = -1
    return tracker


def test_converts_landmarks_and_sides():
    tracker = make_tracker(fake_result(("Left", 0.2), ("Right", 0.7)))
    frame = tracker.process(np.zeros((720, 1280, 3), np.uint8), 1.5)
    assert frame.t == 1.5 and len(frame.hands) == 2
    left = frame.hand(Side.LEFT)
    assert left.landmarks.shape == (21, 3) and left.landmarks[0, 0] == np.float32(0.2)
    assert left.aspect == 1280 / 720 and left.score == np.float32(0.9)
    assert frame.hand(Side.RIGHT).landmarks[0, 0] > 0.6


def test_timestamps_are_strictly_increasing():
    seen = []
    tracker = make_tracker(fake_result())
    tracker._landmarker = SimpleNamespace(detect_for_video=lambda image, ts: (seen.append(ts), fake_result())[1])
    for t in (0.0, 0.0, 0.0004):
        tracker.process(np.zeros((4, 4, 3), np.uint8), t)
    assert seen == sorted(set(seen)) and len(seen) == 3


def test_duplicate_labels_resolved_by_screen_position():
    tracker = make_tracker(fake_result(("Right", 0.8), ("Right", 0.2)))
    frame = tracker.process(np.zeros((720, 1280, 3), np.uint8), 0.0)
    assert frame.hand(Side.LEFT).landmarks[0, 0] < frame.hand(Side.RIGHT).landmarks[0, 0]


def test_resolve_sides_leaves_valid_pairs_alone():
    hands = [make_hand(Side.LEFT), make_hand(Side.RIGHT)]
    assert _resolve_sides(hands) == tuple(hands)
