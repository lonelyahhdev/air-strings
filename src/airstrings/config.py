"""Typed configuration with optional TOML overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class GestureConfig:
    harmony_side: str = "left"  # which hand picks chords; the other hand pinches to play
    # Pinch = thumb tip touching a fingertip, measured in hand-lengths.
    pinch_on: float = 0.30  # closer than this counts as a pinch (raise to make pinching easier)
    pinch_off: float = 0.55  # farther than this lets go
    pinch_margin: float = 0.12  # the pinching fingertip must beat the next-nearest by this much
    pinch_min_reach: float = 0.9  # fingertip must be this far from the wrist (rejects a closed fist)
    pinch_refractory_s: float = 0.07
    # Closing speed (hand-lengths/second) mapped onto soft..loud; min_intensity is the softest hit.
    pinch_slow_speed: float = 1.5
    pinch_fast_speed: float = 8.0
    min_intensity: float = 0.45
    hold_damp_s: float = 0.35  # holding a pinch longer than this damps the sound when released
    settle_s: float = 0.08  # finger count must be stable this long to commit
    # Raising the harmony hand above this line adds a 7th to the chord.
    seventh_zone_y: float = 0.35
    seventh_zone_hysteresis: float = 0.06
    # finger count -> scale degree (I IV V vi ii: the common pop chords first)
    chord_map: dict[int, int] = field(default_factory=lambda: {1: 1, 2: 4, 3: 5, 4: 6, 5: 2})
    switch_hold_s: float = 1.0  # hold both fists this long to swap instrument


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    blocksize: int = 256
    max_voices: int = 48
    master_gain: float = 0.8
    prewarm: bool = True


@dataclass
class AppConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    key: str = "C"
    instrument: str = "guitar"
    mode: str = "free"  # "free" | "sequence"
    model_path: str | None = None  # MediaPipe hand_landmarker.task location


@dataclass
class Config:
    gestures: GestureConfig = field(default_factory=GestureConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    app: AppConfig = field(default_factory=AppConfig)

    @property
    def rhythm_side(self) -> str:
        return "right" if self.gestures.harmony_side == "left" else "left"


def _apply(section: Any, table: dict[str, Any], path: str) -> None:
    valid = {f.name for f in fields(section)}
    for key, value in table.items():
        if key not in valid:
            raise ValueError(f"unknown config option [{path}] {key!r}")
        if key == "chord_map":
            value = {int(k): int(v) for k, v in value.items()}
        setattr(section, key, value)


def load_config(path: str | Path | None = None) -> Config:
    config = Config()
    if path is None:
        return config
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    for name, table in data.items():
        section = getattr(config, name, None)
        if section is None or not is_dataclass(section):
            raise ValueError(f"unknown config section [{name}]")
        _apply(section, table, name)
    if config.gestures.harmony_side not in ("left", "right"):
        raise ValueError("gestures.harmony_side must be 'left' or 'right'")
    return config
