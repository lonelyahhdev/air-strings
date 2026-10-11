"""One performance session: tracking frames in, audio events + UI snapshot out."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from airstrings.audio.events import AudioEvent
from airstrings.config import Config
from airstrings.engine import commands as C
from airstrings.engine.state import Mode, PerformanceState
from airstrings.gestures.chord_selector import ChordSelector
from airstrings.gestures.pinch_detector import PinchDetector
from airstrings.gestures.switch_detector import SwitchDetector
from airstrings.gestures.types import ChordIntent, Hit, Release, Trigger
from airstrings.instruments import instrument_names
from airstrings.music.theory import Key
from airstrings.tracking.types import Hand, Side, TrackingFrame

AudioSink = Callable[[list[AudioEvent]], None]
STATUS_SECONDS = 2.5  # how long a state-change message stays on screen


@dataclass(frozen=True)
class Snapshot:
    """Everything the overlay needs to draw one frame."""

    t: float
    hands: tuple[Hand, ...]
    harmony_side: Side
    instrument: str
    instruments: tuple[str, ...]
    mode: str
    key_name: str
    chord_name: str | None
    chord_ladder: tuple[tuple[int, str], ...]  # (finger count, chord name) for this key
    intent: ChordIntent
    seventh_zone_y: float
    pinch_on: float  # current pinch sensitivity threshold
    pinch_ratios: tuple[float, ...] | None  # thumb-to-fingertip distances (index, middle, ring, pinky)
    pinch_active: Hit | None  # which pinch is currently held
    last_trigger: Trigger | None
    sequence_upcoming: tuple[str, ...]
    sequence_name: str
    switch_progress: float
    status: str = ""


class Session:
    def __init__(self, config: Config, sink: AudioSink):
        self.config = config
        self.sink = sink
        g = config.gestures
        self.harmony_side = Side(g.harmony_side)
        self.rhythm_side = self.harmony_side.other
        self.chord_selector = ChordSelector(g)
        self.pinch = PinchDetector(g)
        self.switch = SwitchDetector(g.switch_hold_s)
        self.state = PerformanceState(
            Key.parse(config.app.key), config.app.instrument, Mode(config.app.mode), hold_damp_s=g.hold_damp_s
        )
        self._last_trigger: Trigger | None = None
        self._intent = ChordIntent(0, None, False, False, False)
        self._status = ""
        self._status_t: float | None = None  # stamped on the next frame
        self.state.subscribe(self._set_status)

    def _set_status(self, message: str) -> None:
        self._status, self._status_t = message, None

    def _emit(self, events: Iterable[AudioEvent]) -> None:
        events = list(events)
        if events:
            self.sink(events)

    def dispatch(self, command: C.Command) -> None:
        """Apply a command from the keyboard (or anywhere else)."""
        if isinstance(command, C.NudgePinch):
            self.pinch.nudge(command.delta)
            self._status, self._status_t = f"pinch sensitivity {self.pinch.on:.2f}", None
            return
        self._emit(self.state.apply(command))

    def process(self, frame: TrackingFrame) -> Snapshot:
        t = frame.t
        if self._status and self._status_t is None:
            self._status_t = t
        harmony = frame.hand(self.harmony_side)
        rhythm = frame.hand(self.rhythm_side)

        self._intent = self.chord_selector.update(harmony, t)
        self._emit(self.state.apply_chord_intent(self._intent))

        if self.switch.update(harmony, rhythm, t):
            self.dispatch(C.CycleInstrument())

        for event in self.pinch.update(rhythm, t):
            if isinstance(event, Trigger):
                self._last_trigger = event
                self._emit(self.state.on_trigger(event))
            elif isinstance(event, Release):
                self._emit(self.state.on_release(event))

        return self._snapshot(frame)

    def _snapshot(self, frame: TrackingFrame) -> Snapshot:
        st = self.state
        chord = st.sounding_chord
        ladder = tuple(
            (fingers, st.key.diatonic_chord(degree, self._intent.extended).name)
            for fingers, degree in sorted(self.config.gestures.chord_map.items())
        )
        return Snapshot(
            t=frame.t,
            hands=frame.hands,
            harmony_side=self.harmony_side,
            instrument=st.instrument.name,
            instruments=tuple(instrument_names()),
            mode=st.mode.value,
            key_name=st.key.name,
            chord_name=chord.name if chord else None,
            chord_ladder=ladder,
            intent=self._intent,
            seventh_zone_y=self.config.gestures.seventh_zone_y,
            pinch_on=self.pinch.on,
            pinch_ratios=self.pinch.ratios,
            pinch_active=self.pinch.active,
            last_trigger=self._last_trigger,
            sequence_upcoming=tuple(st.cursor.upcoming(8)) if st.mode is Mode.SEQUENCE else (),
            sequence_name=st.cursor.sequence.name,
            switch_progress=self.switch.progress,
            status=self._status if self._status_t is not None and frame.t - self._status_t < STATUS_SECONDS else "",
        )
