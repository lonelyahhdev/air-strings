"""Offline note synthesis: each (timbre, pitch, brightness) is rendered once to a buffer.

Rendering whole notes up-front (vectorised NumPy) keeps the realtime audio
callback trivially cheap: it only sums slices of cached buffers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from airstrings.music.theory import midi_to_hz

# render(midi, velocity, sample_rate) -> mono float32, peak ~1
RenderFn = Callable[[int, int, int], np.ndarray]
_REGISTRY: dict[str, RenderFn] = {}


def register_timbre(name: str) -> Callable[[RenderFn], RenderFn]:
    def deco(fn: RenderFn) -> RenderFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def timbre_names() -> list[str]:
    return sorted(_REGISTRY)


def _normalise(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak).astype(np.float32)


@register_timbre("guitar")
def render_guitar(midi: int, velocity: int, sr: int) -> np.ndarray:
    """Karplus-Strong plucked string, vectorised one delay-line period at a time."""
    freq = midi_to_hz(midi)
    n = max(2, int(sr / freq - 0.5))  # delay-line length; loop averager adds 0.5 sample
    actual = sr / (n + 0.5)
    t60 = 3.2 * float(np.clip(2.0 ** (-(midi - 40) / 36.0), 0.35, 1.0))
    rho = 10.0 ** (-3.0 * (n + 0.5) / sr / t60)  # per-period loss for the target T60
    length = int(sr * min(t60, 3.0)) + n

    rng = np.random.default_rng(midi)
    burst = rng.uniform(-1.0, 1.0, n)
    smooth = 1 + int(round(3 * (1 - velocity / 127)))  # soft plucks are duller
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        burst = np.convolve(np.concatenate([burst[-smooth:], burst]), kernel, mode="valid")[-n:]
    pick = max(1, int(n * 0.17))  # pluck-position comb filter
    burst = burst - np.roll(burst, pick)

    y = np.zeros(length + 1)  # y[0] is a zero guard so y[n-1-N] exists at the start
    y[1 : n + 1] = burst
    for start in range(n, length, n):
        end = min(start + n, length)
        a = y[1 + start - n : 1 + end - n]
        b = y[start - n : end - n]
        y[1 + start : 1 + end] = rho * 0.5 * (a + b)
    y = y[1:]

    ratio = freq / actual  # read the buffer slightly faster/slower to land on pitch
    idx = np.arange(int(len(y) / ratio)) * ratio
    y = np.interp(idx, np.arange(len(y)), y)
    fade = min(len(y), int(0.01 * sr))
    y[-fade:] *= np.linspace(1.0, 0.0, fade)
    return _normalise(y)


@register_timbre("piano")
def render_piano(midi: int, velocity: int, sr: int) -> np.ndarray:
    """Additive piano: inharmonic partials, per-partial decay, hammer brightness."""
    f0 = midi_to_hz(midi)
    t60 = float(np.clip(14.0 * 2.0 ** (-(midi - 36) / 30.0), 1.5, 14.0))
    dur = min(t60, 5.0)
    t = np.arange(int(sr * dur)) / sr

    k = np.arange(1, 13)
    freqs = k * f0 * np.sqrt(1.0 + 0.0004 * k**2)
    freqs = freqs[freqs < sr * 0.45]
    k = k[: len(freqs)]
    tilt = 0.9 + 1.6 * (1.0 - velocity / 127.0)  # harder hit -> brighter spectrum
    amps = 1.0 / k**tilt
    amps[1::2] *= 0.8
    decay = (6.9 / t60) * (1.0 + 0.25 * (k - 1))  # 6.9 = ln(1000) -> 60 dB; upper partials die sooner

    # Two strings, the second slightly detuned and quieter: |1 + 0.6 e^{i d t}| is a
    # shallow beating envelope that never reaches zero.  Detune grows per partial.
    t32 = t.astype(np.float32)
    detune = (0.25 * (1.0 + 0.31 * (k - 1)))[:, None].astype(np.float32)
    carrier = np.sin((2 * np.pi * freqs)[:, None].astype(np.float32) * t32)
    beat = np.sqrt(1.36 + 1.2 * np.cos(2 * np.pi * detune * t32)) / 1.6
    env = np.exp(-decay[:, None].astype(np.float32) * t32)
    y = (amps[:, None].astype(np.float32) * carrier * env * beat).sum(axis=0).astype(np.float64)

    attack = int(0.003 * sr)
    y[:attack] *= np.linspace(0.0, 1.0, attack)
    fade = int(0.25 * sr)
    y[-fade:] *= np.linspace(1.0, 0.0, fade)
    return _normalise(y)


class SampleBank:
    """Thread-safe cache of rendered notes, keyed by timbre, pitch and brightness bucket."""

    BUCKETS = 4

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._cache: dict[tuple[str, int, int], np.ndarray] = {}
        self._lock = threading.Lock()

    def get(self, timbre: str, midi: int, velocity: int) -> np.ndarray:
        if timbre not in _REGISTRY:
            raise KeyError(f"unknown timbre {timbre!r}; known: {timbre_names()}")
        bucket = min(self.BUCKETS - 1, max(0, velocity - 1) * self.BUCKETS // 127)
        key = (timbre, midi, bucket)
        buf = self._cache.get(key)
        if buf is None:
            rep_velocity = int((bucket + 0.5) * 127 / self.BUCKETS)
            buf = _REGISTRY[timbre](midi, rep_velocity, self.sample_rate)
            with self._lock:
                self._cache[key] = buf
        return buf

    def prewarm(self, timbre: str, midis: range, velocity: int = 80) -> threading.Thread:
        def work() -> None:
            for m in midis:
                self.get(timbre, m, velocity)

        thread = threading.Thread(target=work, name=f"prewarm-{timbre}", daemon=True)
        thread.start()
        return thread
