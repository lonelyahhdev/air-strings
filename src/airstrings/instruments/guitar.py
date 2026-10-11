"""Strummed guitar: down/up swings sweep across the strings of the chosen chord."""

from __future__ import annotations

from airstrings.audio.events import AudioEvent, NoteOn
from airstrings.gestures.types import Direction, Trigger
from airstrings.instruments.base import Instrument, PlayContext, register_instrument, scaled
from airstrings.music.sequence import SequenceStep, StepKind
from airstrings.music.theory import Chord

LOW_STRING = 40  # E2
STRINGS = 6
MUTED_OPENNESS = 0.3  # below this the strumming hand damps the strings
MUTED_DURATION = 0.18


@register_instrument
class Guitar(Instrument):
    name = "guitar"
    timbre = "guitar"
    midi_range = range(40, 85)

    def _strum(self, chord: Chord, trigger: Trigger, *, soft: float = 1.0) -> list[AudioEvent]:
        notes = chord.stacked(LOW_STRING, STRINGS)
        velocity = trigger.velocity
        if trigger.direction is Direction.UP:  # upstrokes catch the thin strings, a bit lighter
            notes = tuple(reversed(notes[-4:]))
            velocity = scaled(velocity, 0.85)
        spread = 0.045 - 0.035 * trigger.intensity  # harder swing -> tighter strum
        duration = MUTED_DURATION if trigger.openness < MUTED_OPENNESS else None
        return [
            NoteOn(m, scaled(velocity, soft * (0.92 + 0.08 * (i % 2))), self.timbre, delay=i * spread, duration=duration)
            for i, m in enumerate(notes)
        ]

    def on_trigger(self, ctx: PlayContext, trigger: Trigger) -> list[AudioEvent]:
        chord = ctx.chord or ctx.key.diatonic_chord(1)
        # A new strum damps the previous one, like a real strumming hand.
        return self.silence(0.06) + self._strum(chord, trigger)

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
