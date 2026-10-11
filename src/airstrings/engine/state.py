"""The performance state machine.

State: active instrument x mode (free play | sequence) x key x chord.
Inputs: :class:`ChordIntent`, :class:`Trigger` and :class:`Command`.
Output: a list of audio events.  It knows nothing about cameras or sound cards,
and instruments only ever see :class:`PlayContext`.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from airstrings.audio.events import AudioEvent
from airstrings.engine import commands as C
from airstrings.gestures.types import ChordIntent, Release, Trigger
from airstrings.instruments import Instrument, PlayContext, create_instrument, instrument_names
from airstrings.music.harmony import Harmonizer
from airstrings.music.sequence import DEMO_SEQUENCES, Sequence, SequenceCursor, StepKind, parse_sequence
from airstrings.music.theory import Chord, Key


class Mode(Enum):
    FREE = "free"
    SEQUENCE = "sequence"


class PerformanceState:
    def __init__(
        self,
        key: Key,
        instrument: str = "guitar",
        mode: Mode = Mode.FREE,
        auto_harmony: bool = True,
        hold_damp_s: float = 0.35,
    ):
        self.key = key
        self.hold_damp_s = hold_damp_s
        self.mode = mode
        self.auto_harmony = auto_harmony
        self.degree: int | None = 1  # None = chord hand is a fist -> silent
        self.seventh = False
        self.cursor = SequenceCursor(parse_sequence(DEMO_SEQUENCES["beginner"], name="beginner"))
        self.accompaniment: Chord | None = None
        self._harmonizer = Harmonizer(key)
        self._instruments: dict[str, Instrument] = {}
        self.instrument = self._get(instrument)
        self._listeners: list[Callable[[str], None]] = []

    # ---- derived ----------------------------------------------------------------
    @property
    def chord(self) -> Chord | None:
        return None if self.degree is None else self.key.diatonic_chord(self.degree, self.seventh)

    @property
    def sounding_chord(self) -> Chord | None:
        """Chord to show the player: the selected one, or the harmonizer's in sequence mode."""
        return self.chord if self.mode is Mode.FREE else self.accompaniment

    def _ctx(self) -> PlayContext:
        return PlayContext(self.key, self.chord if self.mode is Mode.FREE else self.accompaniment)

    def _get(self, name: str) -> Instrument:
        if name not in self._instruments:
            self._instruments[name] = create_instrument(name)
        return self._instruments[name]

    def subscribe(self, listener: Callable[[str], None]) -> None:
        """Be told (with a short reason string) whenever commands change the state."""
        self._listeners.append(listener)

    def _notify(self, reason: str) -> None:
        for listener in self._listeners:
            listener(reason)

    # ---- gesture inputs --------------------------------------------------------
    def apply_chord_intent(self, intent: ChordIntent) -> list[AudioEvent]:
        if self.mode is not Mode.FREE or not intent.changed:
            return []
        self.degree, self.seventh = intent.degree, intent.extended
        chord = self.chord
        if chord is None:  # fist: cut everything
            return self.instrument.silence(0.15)
        return self.instrument.on_chord_change(self._ctx(), chord)

    def on_trigger(self, trigger: Trigger) -> list[AudioEvent]:
        if self.mode is Mode.FREE:
            if self.degree is None:
                return []
            return self.instrument.on_trigger(self._ctx(), trigger)

        step = self.cursor.next_step()
        if step is None or step.kind is StepKind.REST:
            return []
        events: list[AudioEvent] = []
        if step.kind is StepKind.CHORD:
            self.accompaniment = step.chord
        elif self.auto_harmony:
            chord = self._harmonizer.chord_for_note(step.midi[0], self.accompaniment)
            if chord != self.accompaniment:
                self.accompaniment = chord
                events += self.instrument.on_accompaniment(self._ctx(), chord, trigger)
        return events + self.instrument.on_step(self._ctx(), step, trigger)

    def on_release(self, release: Release) -> list[AudioEvent]:
        """A held pinch opened. Quick taps ring out; a long hold means 'stop' and damps."""
        if release.held_s < self.hold_damp_s or (self.mode is Mode.FREE and self.degree is None):
            return []
        return self.instrument.on_release(self._ctx(), release)

    # ---- commands --------------------------------------------------------------
    def apply(self, command: C.Command) -> list[AudioEvent]:
        match command:
            case C.SwitchInstrument(name):
                return self._switch(name)
            case C.CycleInstrument(step):
                names = instrument_names()
                return self._switch(names[(names.index(self.instrument.name) + step) % len(names)])
            case C.SetMode(mode):
                return self._set_mode(Mode(mode))
            case C.ToggleMode():
                return self._set_mode(Mode.SEQUENCE if self.mode is Mode.FREE else Mode.FREE)
            case C.SetKey(key):
                return self._set_key(key)
            case C.TransposeKey(semitones):
                return self._set_key(self.key.transposed(semitones))
            case C.ToggleKeyMode():
                return self._set_key(self.key.relative_toggle())
            case C.LoadSequence(seq):
                sequence = seq if isinstance(seq, Sequence) else parse_sequence(seq)
                self.cursor = SequenceCursor(sequence)
                self.accompaniment = None
                return self._set_mode(Mode.SEQUENCE)
            case C.ResetSequence():
                self.cursor.reset()
                self.accompaniment = None
                self._notify("sequence reset")
                return []
        return []  # NudgePinch and unknown commands are handled by the session

    def _switch(self, name: str) -> list[AudioEvent]:
        if name == self.instrument.name:
            return []
        new = self._get(name)  # raises KeyError for unknown instruments before any change
        events = self.instrument.deactivate()
        self.instrument = new
        events += new.activate()
        self._notify(f"instrument: {name}")
        return events

    def _set_mode(self, mode: Mode) -> list[AudioEvent]:
        if mode is self.mode:
            return []
        self.mode = mode
        self.accompaniment = None
        self._notify(f"mode: {mode.value}")
        return self.instrument.silence(0.1)

    def _set_key(self, key: Key) -> list[AudioEvent]:
        self.key = key
        self._harmonizer = Harmonizer(key)
        self.accompaniment = None
        self._notify(f"key: {key.name}")
        return self.instrument.silence(0.1)
