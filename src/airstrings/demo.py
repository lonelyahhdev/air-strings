"""Scripted performance driven by synthetic hands: exercises the full pipeline headlessly.

``airstrings demo`` renders it to a WAV; the tests reuse :func:`run_script`.
"""

from __future__ import annotations

from dataclasses import dataclass

from airstrings.audio.backend import OfflineBackend
from airstrings.audio.mixer import Mixer
from airstrings.config import Config
from airstrings.engine import commands as C
from airstrings.engine.session import Session, Snapshot
from airstrings.tracking.synthetic import fingers, make_hand
from airstrings.tracking.types import TrackingFrame

FPS = 30
_FINGER = {"chord": "index", "bass": "middle", "arp": "ring", "hold": "index"}
_TAP = {0: 0.55, 1: 1.0, 2: 1.0, 3: 0.55}  # pinch closing over two frames, then opening


@dataclass(frozen=True)
class Segment:
    seconds: float
    fingers: int = 1  # chord hand finger count
    # One entry per beat: "chord" | "bass" | "arp" (taps), "hold" (index pinch held most of the beat), "rest".
    pattern: tuple[str, ...] = ()
    bpm: float = 120.0
    high: bool = False  # chord hand raised into the 7th zone
    command: C.Command | None = None  # applied at the start of the segment
    harmony_visible: bool = True


def _pinch_amount(token: str, frame_in_beat: int, beat_frames: int) -> float:
    if token == "hold":
        last = max(int(beat_frames * 0.8), 3)
        return 0.55 if frame_in_beat in (0, last) else (1.0 if frame_in_beat < last else 0.0)
    return _TAP.get(frame_in_beat, 0.0)


def run_script(session: Session, backend: OfflineBackend, segments: list[Segment]) -> list[Snapshot]:
    """Feed synthetic frames to ``session`` at 30 fps, rendering audio alongside."""
    cfg = session.config.gestures
    snaps: list[Snapshot] = []
    t = 0.0
    dt = 1.0 / FPS
    for seg in segments:
        if seg.command is not None:
            session.dispatch(seg.command)
        beat_frames = max(int(round(FPS * 60 / seg.bpm)), 5)
        hy = (cfg.seventh_zone_y - 0.12) if seg.high else 0.6
        for i in range(int(seg.seconds * FPS)):
            token = seg.pattern[(i // beat_frames) % len(seg.pattern)] if seg.pattern else "rest"
            amount = _pinch_amount(token, i % beat_frames, beat_frames) if token != "rest" else 0.0
            finger = _FINGER.get(token)
            hands = [make_hand(session.rhythm_side, fingers(5), (0.72, 0.6), pinch=finger, pinch_amount=amount)]
            if seg.harmony_visible:
                hands.append(make_hand(session.harmony_side, fingers(seg.fingers), (0.25, hy)))
            snaps.append(session.process(TrackingFrame(t, tuple(hands))))
            backend.advance(dt)
            t += dt
    return snaps


def default_script() -> list[Segment]:
    guitar = ("bass", "chord", "bass", "chord")
    piano = ("chord", "arp", "bass", "arp")
    return [
        # Guitar: I - IV - V - vi, boom-chick strumming
        Segment(2.0, 1, guitar), Segment(2.0, 2, guitar), Segment(2.0, 3, guitar), Segment(2.0, 4, guitar),
        Segment(2.0, 3, ("arp", "rest", "rest", "rest"), high=True),  # raised hand -> G7, one arpeggio
        # Swap to piano and play the same changes
        Segment(2.0, 1, piano, command=C.SwitchInstrument("piano")), Segment(2.0, 2, piano),
        Segment(2.0, 3, piano), Segment(2.0, 4, ("hold", "rest", "rest", "rest"), bpm=60),  # held chord, then let go
        # Linear sequence: one pinch = one note, auto-harmonised
        Segment(6.0, 1, ("chord",), bpm=100, command=C.LoadSequence("C D D A G E B")),
        Segment(3.0, 1),  # let it ring out
    ]  # fmt: skip


def render_demo(path: str) -> float:
    config = Config()
    mixer = Mixer(config.audio.sample_rate, config.audio.max_voices, config.audio.master_gain)
    backend = OfflineBackend(mixer)
    session = Session(config, backend.submit)
    run_script(session, backend, default_script())
    backend.write_wav(path)
    return len(backend.audio()) / mixer.sample_rate
