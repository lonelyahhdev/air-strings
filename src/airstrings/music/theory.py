"""Pitch, key and chord primitives. Pure Python, no I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass

SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

_LETTER_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b♯♭]?)(\d)?$")

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)

# Tonics whose conventional spelling uses flats.
_FLAT_MAJOR_TONICS = frozenset({5, 10, 3, 8, 1})
_FLAT_MINOR_TONICS = frozenset({2, 7, 0, 5, 10, 3})

QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}
_INTERVALS_TO_QUALITY = {v: k for k, v in QUALITY_INTERVALS.items()}
QUALITY_ALIASES = {"min": "m", "M7": "maj7"}


def parse_note(token: str, default_octave: int = 4) -> int:
    """Parse ``C``, ``f#3``, ``Bb`` ... into a MIDI note number (C4 = 60)."""
    match = _NOTE_RE.match(token)
    if not match:
        raise ValueError(f"not a note name: {token!r}")
    letter, accidental, octave = match.groups()
    semis = _LETTER_SEMITONES[letter.upper()] + _ACCIDENTAL[accidental]
    octave_num = int(octave) if octave is not None else default_octave
    return 12 * (octave_num + 1) + semis


def midi_to_name(midi: int, flats: bool = False) -> str:
    names = FLAT_NAMES if flats else SHARP_NAMES
    return f"{names[midi % 12]}{midi // 12 - 1}"


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


@dataclass(frozen=True)
class Chord:
    root_pc: int
    quality: str = ""
    flats: bool = False

    @property
    def intervals(self) -> tuple[int, ...]:
        return QUALITY_INTERVALS[self.quality]

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple((self.root_pc + i) % 12 for i in self.intervals)

    @property
    def name(self) -> str:
        names = FLAT_NAMES if self.flats else SHARP_NAMES
        return f"{names[self.root_pc]}{self.quality}"

    @classmethod
    def parse(cls, token: str) -> Chord:
        match = re.match(r"^([A-Ga-g])([#b♯♭]?)(.*)$", token)
        if not match:
            raise ValueError(f"not a chord name: {token!r}")
        letter, accidental, quality = match.groups()
        quality = QUALITY_ALIASES.get(quality, quality)
        if quality not in QUALITY_INTERVALS:
            raise ValueError(f"unknown chord quality {quality!r} in {token!r}")
        pc = (_LETTER_SEMITONES[letter.upper()] + _ACCIDENTAL[accidental]) % 12
        return cls(pc, quality, flats=accidental in ("b", "♭"))

    def contains_pc(self, pc: int) -> bool:
        return pc % 12 in self.pitch_classes

    def bass_note(self, low: int = 36) -> int:
        """Lowest root-note MIDI pitch at or above ``low``."""
        return low + (self.root_pc - low) % 12

    def stacked(self, low: int, count: int) -> tuple[int, ...]:
        """Ascending chord tones starting from the root at/above ``low``.

        Cycles through the chord tones until ``count`` notes exist: a guitar-
        style spread voicing (6 notes of C from E2 -> C3 E3 G3 C4 E4 G4).
        """
        pcs = self.pitch_classes
        cur = self.bass_note(low)
        notes = [cur]
        idx = 0
        while len(notes) < count:
            idx = (idx + 1) % len(pcs)
            delta = (pcs[idx] - cur) % 12 or 12
            cur += delta
            notes.append(cur)
        return tuple(notes)

    def close_voicing(self, center: int = 64) -> tuple[int, ...]:
        """Compact voicing (best inversion) whose mean pitch is nearest ``center``."""
        pcs = self.pitch_classes
        best: tuple[float, tuple[int, ...]] | None = None
        for inversion in range(len(pcs)):
            rotated = pcs[inversion:] + pcs[:inversion]
            start = center - 9 + (rotated[0] - (center - 9)) % 12
            notes = [start]
            for pc in rotated[1:]:
                notes.append(notes[-1] + ((pc - notes[-1]) % 12 or 12))
            score = abs(sum(notes) / len(notes) - center)
            if best is None or score < best[0]:
                best = (score, tuple(notes))
        assert best is not None
        return best[1]


@dataclass(frozen=True)
class Key:
    tonic_pc: int
    mode: str = "major"  # "major" | "minor"

    def __post_init__(self) -> None:
        if self.mode not in ("major", "minor"):
            raise ValueError(f"mode must be 'major' or 'minor', got {self.mode!r}")

    @property
    def scale(self) -> tuple[int, ...]:
        return MAJOR_SCALE if self.mode == "major" else MINOR_SCALE

    @property
    def prefer_flats(self) -> bool:
        flat_set = _FLAT_MAJOR_TONICS if self.mode == "major" else _FLAT_MINOR_TONICS
        return self.tonic_pc in flat_set

    @property
    def name(self) -> str:
        names = FLAT_NAMES if self.prefer_flats else SHARP_NAMES
        return f"{names[self.tonic_pc]} {self.mode}"

    @classmethod
    def parse(cls, text: str) -> Key:
        """Accepts ``C``, ``Am``, ``F# minor``, ``Bb major``."""
        match = re.match(r"^\s*([A-Ga-g])([#b♯♭]?)\s*(m|min|minor|maj|major)?\s*$", text)
        if not match:
            raise ValueError(f"cannot parse key: {text!r}")
        letter, accidental, mode = match.groups()
        pc = (_LETTER_SEMITONES[letter.upper()] + _ACCIDENTAL[accidental]) % 12
        return cls(pc, "minor" if mode in ("m", "min", "minor") else "major")

    def transposed(self, semitones: int) -> Key:
        return Key((self.tonic_pc + semitones) % 12, self.mode)

    def relative_toggle(self) -> Key:
        """Switch major <-> minor on the same tonic (parallel key)."""
        return Key(self.tonic_pc, "minor" if self.mode == "major" else "major")

    def contains_pc(self, pc: int) -> bool:
        return (pc - self.tonic_pc) % 12 in self.scale

    def diatonic_chord(self, degree: int, seventh: bool = False) -> Chord:
        """Chord built by stacking scale thirds on ``degree`` (1-based, 1..7)."""
        if not 1 <= degree <= 7:
            raise ValueError(f"degree must be 1..7, got {degree}")
        scale = self.scale

        def tone(k: int) -> int:
            i = degree - 1 + k
            return scale[i % 7] + 12 * (i // 7)

        root = tone(0)
        steps = (0, 2, 4, 6) if seventh else (0, 2, 4)
        intervals = tuple(tone(k) - root for k in steps)
        quality = _INTERVALS_TO_QUALITY.get(intervals)
        if quality is None:  # never hit for natural major/minor, but stay safe
            quality = _INTERVALS_TO_QUALITY[intervals[:3]]
        return Chord((self.tonic_pc + root) % 12, quality, self.prefer_flats)
