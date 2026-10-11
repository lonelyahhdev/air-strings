"""Typed configuration with optional TOML overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class GestureConfig:
    harmony_side: str = "left"  # which hand picks chords; the other one plays
    # Strum line, as a fraction of image height (0 = top).
    strum_line_y: float = 0.55
    # Palm speed, in hand-lengths per second, mapped onto soft..loud.
    min_strum_speed: float = 3.0
    full_strum_speed: float = 16.0
    refractory_s: float = 0.09
    lost_gap_s: float = 0.25  # a gap this long resets motion tracking
    settle_s: float = 0.08  # finger count must be stable this long to commit
    # Raising the harmony hand above this line adds a 7th to the chord.
    seventh_zone_y: float = 0.35
    seventh_zone_hysteresis: float = 0.06
    # finger count -> scale degree (I IV V vi ii: the common pop chords first)
    chord_map: dict[int, int] = field(default_factory=lambda: {1: 1, 2: 4, 3: 5, 4: 6, 5: 2})
    switch_hold_s: float = 1.0  # hold both fists this long to swap instrument
    smoothing_min_cutoff: float = 2.0
    smoothing_beta: float = 6.0


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
