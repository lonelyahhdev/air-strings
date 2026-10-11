"""Everything that can change the performance state, from gestures *or* keyboard."""

from __future__ import annotations

from dataclasses import dataclass

from airstrings.music.sequence import Sequence
from airstrings.music.theory import Key


class Command:
    """Marker base class."""


@dataclass(frozen=True)
class SwitchInstrument(Command):
    name: str


@dataclass(frozen=True)
class CycleInstrument(Command):
    step: int = 1


@dataclass(frozen=True)
class SetMode(Command):
    mode: str  # "free" | "sequence"


@dataclass(frozen=True)
class ToggleMode(Command):
    pass


@dataclass(frozen=True)
class SetKey(Command):
    key: Key


@dataclass(frozen=True)
class TransposeKey(Command):
    semitones: int


@dataclass(frozen=True)
class ToggleKeyMode(Command):
    pass


@dataclass(frozen=True)
class LoadSequence(Command):
    sequence: Sequence | str  # text is parsed with the default notation rules


@dataclass(frozen=True)
class ResetSequence(Command):
    pass


@dataclass(frozen=True)
class NudgeStrumLine(Command):
    delta: float
