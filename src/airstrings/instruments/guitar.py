"""Guitar: pinch index = strum the chord, middle = bass note, ring = picked arpeggio."""

from __future__ import annotations

from airstrings.audio.events import AudioEvent, NoteOn
from airstrings.gestures.types import Hit, Trigger
from airstrings.instruments.base import Instrument, PlayContext, register_instrument, scaled
from airstrings.music.sequence import SequenceStep, StepKind
from airstrings.music.theory import Chord

LOW_STRING = 40  # E2
STRINGS = 6
ARP_STEP = 0.075  # seconds between picked strings


@register_instrument
class Guitar(Instrument):
    name = "guitar"
    timbre = "guitar"
    midi_range = range(40, 85)
    damp_fade = 0.25

    def __init__(self) -> None:
        self._bass_on_fifth = False

    def _strum(self, chord: Chord, trigger: Trigger, *, soft: float = 1.0) -> list[AudioEvent]:
        notes = chord.stacked(LOW_STRING, STRINGS)
        spread = 0.045 - 0.035 * trigger.intensity  # harder hit -> tighter strum
        return [
            NoteOn(m, scaled(trigger.velocity, soft * (0.92 + 0.08 * (i % 2))), self.timbre, delay=i * spread)
            for i, m in enumerate(notes)
        ]

    def _bass(self, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        """Alternates root and fifth, the classic strummed-guitar bass line."""
        root = chord.bass_note(LOW_STRING)
        note = root + 7 if self._bass_on_fifth else root
        self._bass_on_fifth = not self._bass_on_fifth
        return [NoteOn(note, trigger.velocity, self.timbre)]

    def _arpeggio(self, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        notes = chord.stacked(LOW_STRING, STRINGS)
        return [
            NoteOn(m, scaled(trigger.velocity, 0.85), self.timbre, delay=i * ARP_STEP)
            for i, m in enumerate(notes)
        ]

    def on_chord_change(self, ctx: PlayContext, chord: Chord) -> list[AudioEvent]:
        self._bass_on_fifth = False  # a new chord starts its bass pattern on the root
        return []

    def on_trigger(self, ctx: PlayContext, trigger: Trigger) -> list[AudioEvent]:
        chord = ctx.chord or ctx.key.diatonic_chord(1)
        if trigger.kind is Hit.BASS:
            return self._bass(chord, trigger)
        # A new strum or arpeggio damps the previous one, like a real strumming hand.
        play = self._arpeggio if trigger.kind is Hit.ARP else self._strum
        return self.silence(0.06) + play(chord, trigger)

    def on_step(self, ctx: PlayContext, step: SequenceStep, trigger: Trigger) -> list[AudioEvent]:
        if step.kind is StepKind.REST:
            return []
        if step.kind is StepKind.CHORD and step.chord is not None:
            return self.silence(0.06) + self._strum(step.chord, trigger)
        # Single plucked notes ring over each other, like fingerpicking.
        return [NoteOn(m, trigger.velocity, self.timbre) for m in step.midi]

    def on_accompaniment(self, ctx: PlayContext, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        notes = chord.stacked(LOW_STRING, 4)
        return [
            NoteOn(m, scaled(trigger.velocity, 0.45), self.timbre, delay=0.02 + i * 0.03, duration=1.4)
            for i, m in enumerate(notes)
        ]
