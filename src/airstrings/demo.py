"""Scripted performance driven by synthetic hands: exercises the full pipeline headlessly.

``airstrings demo`` renders it to a WAV; the tests reuse :func:`run_script`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from airstrings.audio.backend import OfflineBackend
from airstrings.audio.mixer import Mixer
from airstrings.config import Config
from airstrings.engine import commands as C
from airstrings.engine.session import Session, Snapshot
from airstrings.tracking.synthetic import fingers, make_hand
from airstrings.tracking.types import TrackingFrame

FPS = 30


@dataclass(frozen=True)
class Segment:
    seconds: float
    fingers: int = 1  # chord hand finger count
    strum_hz: float = 2.0  # full swings (down+up) per second; 0 = hand rests
    high: bool = False  # chord hand raised into the 7th zone
    command: C.Command | None = None  # applied at the start of the segment
    harmony_visible: bool = True
    swing_amplitude: float = 0.14


def run_script(session: Session, backend: OfflineBackend, segments: list[Segment]) -> list[Snapshot]:
    """Feed synthetic frames to ``session`` at 30 fps, rendering audio alongside."""
    cfg = session.config.gestures
    snaps: list[Snapshot] = []
    t = 0.0
    dt = 1.0 / FPS
    for seg in segments:
        if seg.command is not None:
            session.dispatch(seg.command)
        for i in range(int(seg.seconds * FPS)):
            phase = 2 * math.pi * seg.strum_hz * (i * dt)
            # start moving upward-then-down so a downstroke is the first crossing
            ry = cfg.strum_line_y - seg.swing_amplitude * math.sin(phase) if seg.strum_hz else cfg.strum_line_y - 0.2
            hy = (cfg.seventh_zone_y - 0.12) if seg.high else 0.6
            hands = [make_hand(session.rhythm_side, fingers(5), (0.72, ry))]
            if seg.harmony_visible:
                hands.append(make_hand(session.harmony_side, fingers(seg.fingers), (0.25, hy)))
            snaps.append(session.process(TrackingFrame(t, tuple(hands))))
            backend.advance(dt)
            t += dt
    return snaps


def default_script() -> list[Segment]:
    return [
        # Guitar: I - IV - V - vi, strumming
        Segment(2.0, 1), Segment(2.0, 2), Segment(2.0, 3), Segment(2.0, 4),
        Segment(2.0, 3, high=True),  # raised hand -> G7
        # Swap to piano and play the same changes
        Segment(2.0, 1, command=C.SwitchInstrument("piano")), Segment(2.0, 2), Segment(2.0, 3), Segment(2.0, 4),
        # Linear sequence: one swing = one note, auto-harmonised
        Segment(6.0, 1, strum_hz=1.5, command=C.LoadSequence("C D D A G E B")),
        Segment(3.0, 1, strum_hz=0.0),  # let it ring out
    ]  # fmt: skip


def render_demo(path: str) -> float:
    config = Config()
    mixer = Mixer(config.audio.sample_rate, config.audio.max_voices, config.audio.master_gain)
    backend = OfflineBackend(mixer)
    session = Session(config, backend.submit)
    run_script(session, backend, default_script())
    backend.write_wav(path)
    return len(backend.audio()) / mixer.sample_rate
