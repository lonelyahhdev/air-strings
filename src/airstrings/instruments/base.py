"""Instrument contract and registry.

An instrument turns *musical intent* (a trigger, a chord, a sequence step) into
:mod:`airstrings.audio.events`.  It never touches cameras, landmarks or audio
devices, so new instruments are drop-in: subclass, decorate, done.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from airstrings.audio.events import AudioEvent, ReleaseAll
from airstrings.gestures.types import Trigger
from airstrings.music.sequence import SequenceStep
from airstrings.music.theory import Chord, Key


@dataclass(frozen=True)
class PlayContext:
    key: Key
    chord: Chord | None  # chord the player has selected (free play) or None


class Instrument(ABC):
    name: ClassVar[str]  # registry id, also shown in the UI
    timbre: ClassVar[str]  # synth registry id
    midi_range: ClassVar[range] = range(36, 85)  # used to prewarm the sample cache

    @abstractmethod
    def on_trigger(self, ctx: PlayContext, trigger: Trigger) -> list[AudioEvent]:
        """Free play: the playing hand hit; sound the selected chord."""

    @abstractmethod
    def on_step(self, ctx: PlayContext, step: SequenceStep, trigger: Trigger) -> list[AudioEvent]:
        """Sequence mode: sound one step (note(s) or chord) of the linear sequence."""

    @abstractmethod
    def on_accompaniment(self, ctx: PlayContext, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        """Soft backing for an auto-harmonised melody; called when the chord changes."""

    def on_chord_change(self, ctx: PlayContext, chord: Chord) -> list[AudioEvent]:
        """Chord hand changed chord (free play). Default: nothing until the next hit."""
        return []

    def silence(self, fade: float = 0.15) -> list[AudioEvent]:
        return [ReleaseAll(self.timbre, fade)]

    def activate(self) -> list[AudioEvent]:
        return []

    def deactivate(self) -> list[AudioEvent]:
        return self.silence(0.1)


_REGISTRY: dict[str, type[Instrument]] = {}


def register_instrument(cls: type[Instrument]) -> type[Instrument]:
    _REGISTRY[cls.name] = cls
    return cls


def instrument_names() -> list[str]:
    return list(_REGISTRY)


def create_instrument(name: str) -> Instrument:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown instrument {name!r}; available: {instrument_names()}") from None


def scaled(velocity: int, factor: float) -> int:
    """Scale a MIDI velocity, keeping it in 1..127."""
    return max(1, min(127, int(round(velocity * factor))))
