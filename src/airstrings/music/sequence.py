"""Beginner-friendly linear sequences.

A sequence is a list of steps the performer walks through one gesture at a
time. Three notations are accepted, and can be mixed:

* note names      ``C D D A G E B``  or  ``C-D-D-A-G-E-B``  (``C5`` sets octave)
* guitar frets    ``0 2 3 5``  (default string)  or  ``5:3`` (string 5, fret 3)
* chord names     ``Am F C G``  (a bare letter is a chord only when the
  sequence also contains a chord with a suffix, e.g. ``C G Am F``)

``-``, ``.``, ``_`` or ``~`` on their own are rests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from airstrings.music.theory import Chord, QUALITY_INTERVALS, parse_note

# MIDI pitches of the open strings, string 6 (low E) first.
GUITAR_TUNING = (40, 45, 50, 55, 59, 64)

REST_TOKENS = frozenset({"-", ".", "_", "~"})
_FRET_RE = re.compile(r"^(?:([1-6]):)?(\d{1,2})$")
_QUALITIES = sorted((q for q in QUALITY_INTERVALS if q), key=len, reverse=True)
_QUALITY_ALIASES = ["min", "M7"]
_CHORD_SUFFIX_RE = re.compile(
    rf"^([A-Ga-g])([#b]?)({'|'.join(map(re.escape, _QUALITIES + _QUALITY_ALIASES))})$"
)


class StepKind(Enum):
    NOTE = "note"
    CHORD = "chord"
    REST = "rest"


@dataclass(frozen=True)
class SequenceStep:
    kind: StepKind
    label: str
    midi: tuple[int, ...] = ()
    chord: Chord | None = None


REST_STEP = SequenceStep(StepKind.REST, "-")


def fret_to_midi(string: int, fret: int) -> int:
    """String 6 = low E ... string 1 = high E."""
    if not 1 <= string <= 6:
        raise ValueError(f"guitar string must be 1..6, got {string}")
    return GUITAR_TUNING[6 - string] + fret


@dataclass(frozen=True)
class Sequence:
    steps: tuple[SequenceStep, ...]
    loop: bool = True
    name: str = ""

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class SequenceCursor:
    """Walks a :class:`Sequence`; ``index`` is the next step to be played."""

    sequence: Sequence
    index: int = 0
    finished: bool = field(default=False, init=False)

    def next_step(self) -> SequenceStep | None:
        if self.finished or not self.sequence.steps:
            return None
        step = self.sequence.steps[self.index]
        self.index += 1
        if self.index >= len(self.sequence.steps):
            if self.sequence.loop:
                self.index = 0
            else:
                self.finished = True
        return step

    def reset(self) -> None:
        self.index = 0
        self.finished = False

    def upcoming(self, count: int) -> list[str]:
        """Labels of the next ``count`` steps (wrapping if the sequence loops)."""
        steps = self.sequence.steps
        if self.finished or not steps:
            return []
        labels = []
        for k in range(count):
            i = self.index + k
            if i >= len(steps):
                if not self.sequence.loop:
                    break
                i %= len(steps)
            labels.append(steps[i].label)
        return labels


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for part in re.split(r"[\s,;|]+", text.strip()):
        if not part:
            continue
        if part in REST_TOKENS:
            tokens.append(part)
        else:  # "C-D-D-A" style: hyphens separate notes
            tokens.extend(p for p in part.split("-") if p)
    return tokens


def parse_sequence(
    text: str,
    *,
    default_octave: int = 4,
    default_string: int = 4,
    loop: bool = True,
    name: str = "",
) -> Sequence:
    tokens = _tokenize(text)
    if not tokens:
        raise ValueError("sequence is empty")
    chord_mode = any(_CHORD_SUFFIX_RE.match(t) for t in tokens)

    steps: list[SequenceStep] = []
    for token in tokens:
        if token in REST_TOKENS:
            steps.append(REST_STEP)
            continue
        fret = _FRET_RE.match(token)
        if fret:
            string = int(fret.group(1)) if fret.group(1) else default_string
            midi = fret_to_midi(string, int(fret.group(2)))
            steps.append(SequenceStep(StepKind.NOTE, token, (midi,)))
            continue
        is_chord = bool(_CHORD_SUFFIX_RE.match(token)) or (
            chord_mode and re.fullmatch(r"[A-Ga-g][#b]?", token) is not None
        )
        try:
            if is_chord:
                chord = Chord.parse(token)
                steps.append(SequenceStep(StepKind.CHORD, chord.name, (), chord))
            else:
                midi = parse_note(token, default_octave)
                steps.append(SequenceStep(StepKind.NOTE, token, (midi,)))
        except ValueError as exc:
            raise ValueError(f"cannot read {token!r} in sequence: {exc}") from exc
    return Sequence(tuple(steps), loop=loop, name=name)


DEMO_SEQUENCES = {
    "beginner": "C D D A G E B",
    "scale": "C D E F G A B C5",
    "twinkle": "C C G G A A G - F F E E D D C",
    "frets": "0 2 3 5 3 2 0 -",
    "pop": "C G Am F",
}
