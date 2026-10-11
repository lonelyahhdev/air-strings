"""Audio output backends. All share the Mixer; only the sink differs."""

from __future__ import annotations

import wave
from typing import Protocol

import numpy as np

from airstrings.audio.events import AudioEvent
from airstrings.audio.mixer import Mixer


class AudioBackend(Protocol):
    mixer: Mixer

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def submit(self, events: list[AudioEvent]) -> None: ...


class SoundDeviceBackend:
    """Realtime output through PortAudio (the ``sounddevice`` package)."""

    def __init__(self, mixer: Mixer, blocksize: int = 256, device: int | str | None = None):
        self.mixer = mixer
        self.blocksize = blocksize
        self.device = device
        self._stream = None

    def start(self) -> None:
        try:
            import sounddevice as sd
        except OSError as exc:  # PortAudio shared library missing
            raise RuntimeError(
                "PortAudio not found. Install it (Debian/Ubuntu: `sudo apt install libportaudio2`, "
                "macOS: `brew install portaudio`) or run with --no-audio."
            ) from exc
        self._stream = sd.OutputStream(
            samplerate=self.mixer.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            latency="low",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, _time, _status) -> None:
        outdata[:, 0] = self.mixer.render(frames)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def submit(self, events: list[AudioEvent]) -> None:
        self.mixer.submit(events)


class OfflineBackend:
    """Renders to memory instead of a device: tests, demos, WAV export."""

    def __init__(self, mixer: Mixer):
        self.mixer = mixer
        self._chunks: list[np.ndarray] = []

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def submit(self, events: list[AudioEvent]) -> None:
        self.mixer.submit(events)

    def advance(self, seconds: float) -> np.ndarray:
        """Render ``seconds`` of audio (as the device would during that time)."""
        frames = int(round(seconds * self.mixer.sample_rate))
        chunk = self.mixer.render(frames)
        self._chunks.append(chunk)
        return chunk

    def audio(self) -> np.ndarray:
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)

    def write_wav(self, path: str) -> None:
        pcm = (np.clip(self.audio(), -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(self.mixer.sample_rate)
            fh.writeframes(pcm.tobytes())
