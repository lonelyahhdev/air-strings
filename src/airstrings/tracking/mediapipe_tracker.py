"""MediaPipe Hand Landmarker wrapper (Tasks API, video mode)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

from airstrings.tracking.types import Hand, Side, TrackingFrame

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def default_model_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "airstrings" / "hand_landmarker.task"


def ensure_model(path: str | Path | None = None) -> Path:
    """Return the model file, downloading it on first use."""
    target = Path(path) if path else default_model_path()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".part")
        print(f"Downloading hand model to {target} ...")
        urllib.request.urlretrieve(MODEL_URL, part)
        part.replace(target)
    return target


class MediaPipeHandTracker:
    """Feed it *mirrored* RGB frames; handedness labels then match the user's real hands."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_hands: int = 2,
        min_confidence: float = 0.5,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, rgb: np.ndarray, t: float) -> TrackingFrame:
        height, width = rgb.shape[:2]
        # Timestamps must strictly increase.
        ts = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)

        hands: list[Hand] = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            side = Side.LEFT if handed[0].category_name == "Left" else Side.RIGHT
            hands.append(Hand(side, arr, float(handed[0].score), width / height))
        return TrackingFrame(t, _resolve_sides(hands))

    def close(self) -> None:
        self._landmarker.close()


def _resolve_sides(hands: list[Hand]) -> tuple[Hand, ...]:
    """If both hands got the same label, assign by screen position (mirrored view)."""
    if len(hands) == 2 and hands[0].side is hands[1].side:
        ordered = sorted(hands, key=lambda h: float(h.landmarks[0, 0]))
        hands = [
            Hand(Side.LEFT, ordered[0].landmarks, ordered[0].score, ordered[0].aspect),
            Hand(Side.RIGHT, ordered[1].landmarks, ordered[1].score, ordered[1].aspect),
        ]
    return tuple(hands)
