"""Piano: a hit plays bass + chord; upstrokes roll the chord upward like an arpeggio."""

from __future__ import annotations

from airstrings.audio.events import AudioEvent, NoteOn
from airstrings.gestures.types import Direction, Trigger
from airstrings.instruments.base import Instrument, PlayContext, register_instrument, scaled
from airstrings.music.sequence import SequenceStep, StepKind
from airstrings.music.theory import Chord

BASS_LOW = 36  # C2
VOICING_CENTER = 64
STACCATO_OPENNESS = 0.3
STACCATO_DURATION = 0.25


@register_instrument
class Piano(Instrument):
    name = "piano"
    timbre = "piano"
    midi_range = range(36, 85)

    def _duration(self, trigger: Trigger) -> float | None:
        return STACCATO_DURATION if trigger.openness < STACCATO_OPENNESS else None

    def _comp(self, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        voicing = chord.close_voicing(VOICING_CENTER)
        duration = self._duration(trigger)
        v = trigger.velocity
        if trigger.direction is Direction.DOWN:
            events: list[AudioEvent] = [NoteOn(chord.bass_note(BASS_LOW), scaled(v, 0.9), self.timbre, duration=duration)]
            events += [NoteOn(m, scaled(v, 0.85), self.timbre, delay=0.004 * i, duration=duration) for i, m in enumerate(voicing)]
            return events
        rolled = voicing + (voicing[0] + 12,)
        return [
            NoteOn(m, scaled(v, 0.75 + 0.05 * i), self.timbre, delay=0.05 * i, duration=duration)
            for i, m in enumerate(rolled)
        ]

    def on_trigger(self, ctx: PlayContext, trigger: Trigger) -> list[AudioEvent]:
        chord = ctx.chord or ctx.key.diatonic_chord(1)
        return self.silence(0.25) + self._comp(chord, trigger)

    def on_step(self, ctx: PlayContext, step: SequenceStep, trigger: Trigger) -> list[AudioEvent]:
        if step.kind is StepKind.REST:
            return []
        if step.kind is StepKind.CHORD and step.chord is not None:
            return self.silence(0.25) + self._comp(step.chord, trigger)
        return [NoteOn(m, trigger.velocity, self.timbre, duration=self._duration(trigger)) for m in step.midi]

    def on_accompaniment(self, ctx: PlayContext, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        pad = chord.close_voicing(55)
        events = [NoteOn(chord.bass_note(BASS_LOW), scaled(trigger.velocity, 0.5), self.timbre, duration=1.8)]
        events += [NoteOn(m, scaled(trigger.velocity, 0.35), self.timbre, delay=0.01 * i, duration=1.8) for i, m in enumerate(pad)]
        return events
