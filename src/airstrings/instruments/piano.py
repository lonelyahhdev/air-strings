"""Piano: pinch index = bass + chord, middle = bass note, ring = rolled arpeggio."""

from __future__ import annotations

from airstrings.audio.events import AudioEvent, NoteOn
from airstrings.gestures.types import Hit, Trigger
from airstrings.instruments.base import Instrument, PlayContext, register_instrument, scaled
from airstrings.music.sequence import SequenceStep, StepKind
from airstrings.music.theory import Chord

BASS_LOW = 36  # C2
VOICING_CENTER = 64
ARP_STEP = 0.07


@register_instrument
class Piano(Instrument):
    name = "piano"
    timbre = "piano"
    midi_range = range(36, 85)
    damp_fade = 0.4

    def __init__(self) -> None:
        self._bass_on_fifth = False

    def _block(self, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        v = trigger.velocity
        events: list[AudioEvent] = [NoteOn(chord.bass_note(BASS_LOW), scaled(v, 0.9), self.timbre)]
        for i, m in enumerate(chord.close_voicing(VOICING_CENTER)):
            events.append(NoteOn(m, scaled(v, 0.85), self.timbre, delay=0.004 * i))
        return events

    def _bass(self, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        root = chord.bass_note(BASS_LOW)
        note = root + 7 if self._bass_on_fifth else root
        self._bass_on_fifth = not self._bass_on_fifth
        return [NoteOn(note, trigger.velocity, self.timbre)]

    def _arpeggio(self, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        voicing = chord.close_voicing(VOICING_CENTER)
        rolled = (chord.bass_note(BASS_LOW),) + voicing + (voicing[0] + 12,)
        return [
            NoteOn(m, scaled(trigger.velocity, 0.75 + 0.04 * i), self.timbre, delay=ARP_STEP * i)
            for i, m in enumerate(rolled)
        ]

    def on_chord_change(self, ctx: PlayContext, chord: Chord) -> list[AudioEvent]:
        self._bass_on_fifth = False
        return []

    def on_trigger(self, ctx: PlayContext, trigger: Trigger) -> list[AudioEvent]:
        chord = ctx.chord or ctx.key.diatonic_chord(1)
        if trigger.kind is Hit.BASS:
            return self._bass(chord, trigger)
        play = self._arpeggio if trigger.kind is Hit.ARP else self._block
        return self.silence(0.25) + play(chord, trigger)

    def on_step(self, ctx: PlayContext, step: SequenceStep, trigger: Trigger) -> list[AudioEvent]:
        if step.kind is StepKind.REST:
            return []
        if step.kind is StepKind.CHORD and step.chord is not None:
            return self.silence(0.25) + self._block(step.chord, trigger)
        return [NoteOn(m, trigger.velocity, self.timbre) for m in step.midi]

    def on_accompaniment(self, ctx: PlayContext, chord: Chord, trigger: Trigger) -> list[AudioEvent]:
        pad = chord.close_voicing(55)
        events = [NoteOn(chord.bass_note(BASS_LOW), scaled(trigger.velocity, 0.5), self.timbre, duration=1.8)]
        events += [NoteOn(m, scaled(trigger.velocity, 0.35), self.timbre, delay=0.01 * i, duration=1.8) for i, m in enumerate(pad)]
        return events
