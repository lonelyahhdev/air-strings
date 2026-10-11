"""Polyphonic mixer: sample-accurate scheduling, releases, choking, voice stealing.

``submit`` may be called from any thread (it resolves sample buffers up-front);
``render`` is the realtime-safe callback body and is also used for offline
rendering, so the whole audio path is testable without a sound card.
"""

from __future__ import annotations

import queue

import numpy as np

from airstrings.audio.events import AudioEvent, NoteOn, ReleaseAll
from airstrings.audio.synth import SampleBank

CHOKE_FADE_S = 0.03  # same pitch re-struck: previous voice fades this fast
STEAL_FADE_S = 0.015


class _Voice:
    __slots__ = ("buf", "gain", "start", "pos", "fade_start", "fade_len", "timbre", "midi", "dead")

    def __init__(self, buf: np.ndarray, gain: float, start: int, fade_len: int, timbre: str, midi: int):
        self.buf = buf
        self.gain = gain
        self.start = start  # absolute frame the first sample sounds
        self.pos = 0
        self.fade_start: int | None = None  # absolute frame the fade-out begins
        self.fade_len = max(1, fade_len)
        self.timbre = timbre
        self.midi = midi
        self.dead = False

    def release_at(self, frame: int, fade_len: int | None = None) -> None:
        if fade_len is not None:
            self.fade_len = max(1, min(self.fade_len, fade_len)) if self.fade_start is not None else max(1, fade_len)
        if self.fade_start is None or frame < self.fade_start:
            self.fade_start = frame
        if self.fade_start <= self.start:
            self.dead = True  # released before it ever sounded


class Mixer:
    def __init__(self, sample_rate: int = 44100, max_voices: int = 48, master_gain: float = 0.8):
        self.sample_rate = sample_rate
        self.max_voices = max_voices
        self.master_gain = master_gain
        self.bank = SampleBank(sample_rate)
        self._inbox: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self._voices: list[_Voice] = []
        self._clock = 0  # frames rendered so far

    @property
    def active_voices(self) -> int:
        return sum(1 for v in self._voices if not v.dead)

    # ---- producer side (any thread) -------------------------------------------------
    def submit(self, events: list[AudioEvent] | tuple[AudioEvent, ...]) -> None:
        for event in events:
            if isinstance(event, NoteOn):
                buf = self.bank.get(event.timbre, event.midi, event.velocity)
                self._inbox.put(("note", (event, buf)))
            else:
                self._inbox.put(("release", event))

    # ---- consumer side (audio thread) -----------------------------------------------
    def _secs(self, seconds: float) -> int:
        return int(round(seconds * self.sample_rate))

    def _drain(self) -> None:
        while True:
            try:
                kind, payload = self._inbox.get_nowait()
            except queue.Empty:
                return
            if kind == "note":
                self._start_note(*payload)  # type: ignore[misc]
            else:
                self._release_all(payload)  # type: ignore[arg-type]

    def _start_note(self, event: NoteOn, buf: np.ndarray) -> None:
        start = self._clock + self._secs(event.delay)
        gain = (event.velocity / 127.0) ** 1.3 * 0.45
        voice = _Voice(buf, gain, start, self._secs(event.release), event.timbre, event.midi)
        if event.duration is not None:
            voice.release_at(start + self._secs(event.duration))
        for old in self._voices:  # re-striking a pitch chokes the previous voice
            if old.timbre == event.timbre and old.midi == event.midi and not old.dead:
                old.release_at(start, self._secs(CHOKE_FADE_S))
        self._voices.append(voice)
        live = [v for v in self._voices if not v.dead and v.fade_start is None]
        for old in live[: max(0, len(live) - self.max_voices)]:  # steal oldest
            old.release_at(self._clock, self._secs(STEAL_FADE_S))

    def _release_all(self, event: ReleaseAll) -> None:
        fade = self._secs(event.fade)
        for voice in self._voices:
            if voice.dead or (event.timbre is not None and voice.timbre != event.timbre):
                continue
            voice.release_at(self._clock, fade)

    def render(self, frames: int) -> np.ndarray:
        self._drain()
        t0, t1 = self._clock, self._clock + frames
        out = np.zeros(frames, dtype=np.float32)
        for v in self._voices:
            if v.dead:
                continue
            off = max(0, v.start - t0)
            if off >= frames:
                continue  # scheduled for a later block
            n = min(frames - off, len(v.buf) - v.pos)
            if n <= 0:
                v.dead = True
                continue
            seg = v.buf[v.pos : v.pos + n] * v.gain
            if v.fade_start is not None and v.fade_start < t0 + off + n:
                idx = np.arange(t0 + off, t0 + off + n)
                seg = seg * np.clip(1.0 - (idx - v.fade_start) / v.fade_len, 0.0, 1.0)
                if t0 + off + n >= v.fade_start + v.fade_len:
                    v.dead = True
            out[off : off + n] += seg
            v.pos += n
            if v.pos >= len(v.buf):
                v.dead = True
        self._voices = [v for v in self._voices if not v.dead]
        self._clock = t1
        return np.tanh(out * self.master_gain).astype(np.float32)
