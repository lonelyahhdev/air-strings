"""Backend-agnostic audio commands. Instruments emit these; the mixer consumes them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoteOn:
    midi: int
    velocity: int  # 1..127
    timbre: str  # key into the synth registry, e.g. "guitar"
    delay: float = 0.0  # seconds from "now" (sample-accurate scheduling)
    duration: float | None = None  # None = let it ring/decay naturally
    release: float = 0.12  # fade-out time (s) once released


@dataclass(frozen=True)
class ReleaseAll:
    timbre: str | None = None  # None = every timbre
    fade: float = 0.08


AudioEvent = NoteOn | ReleaseAll
