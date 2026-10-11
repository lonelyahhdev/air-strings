"""Realtime application loop: camera -> tracker -> session -> overlay + audio."""

from __future__ import annotations

import time

from airstrings.audio.backend import AudioBackend, OfflineBackend, SoundDeviceBackend
from airstrings.audio.mixer import Mixer
from airstrings.config import Config
from airstrings.engine import commands as C
from airstrings.engine.session import Session
from airstrings.instruments import instrument_names

KEY_HELP = "q quit | 1-9 instrument | tab next | m mode | r restart seq | , . transpose | n major/minor | [ ] line"


def key_to_command(key: int) -> C.Command | None:
    """Map a cv2.waitKey code to a command. Pure, so it is unit-testable."""
    ch = chr(key) if 0 <= key < 128 else ""
    names = instrument_names()
    if ch.isdigit() and ch != "0" and int(ch) <= len(names):
        return C.SwitchInstrument(names[int(ch) - 1])
    return {
        "\t": C.CycleInstrument(),
        "m": C.ToggleMode(),
        "r": C.ResetSequence(),
        ",": C.TransposeKey(-1),
        ".": C.TransposeKey(1),
        "n": C.ToggleKeyMode(),
        "[": C.NudgeStrumLine(-0.02),
        "]": C.NudgeStrumLine(0.02),
    }.get(ch)


def run(config: Config, *, sequence: str | None = None, audio: bool = True) -> None:
    import cv2

    from airstrings.tracking.mediapipe_tracker import MediaPipeHandTracker
    from airstrings.ui.overlay import Overlay

    mixer = Mixer(config.audio.sample_rate, config.audio.max_voices, config.audio.master_gain)
    backend: AudioBackend = SoundDeviceBackend(mixer, config.audio.blocksize) if audio else OfflineBackend(mixer)
    session = Session(config, sink=backend.submit)
    if sequence:
        session.dispatch(C.LoadSequence(sequence))
    if config.audio.prewarm:
        for name in instrument_names():
            inst = session.state._get(name)  # warm every instrument's sample cache in the background
            mixer.bank.prewarm(inst.timbre, inst.midi_range)

    tracker = MediaPipeHandTracker(config.app.model_path)
    cap = cv2.VideoCapture(config.app.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.app.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.app.height)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {config.app.camera_index}")

    overlay = Overlay()
    backend.start()
    print(KEY_HELP)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # mirror: hands move like in a mirror
            t = time.monotonic()
            snap = session.process(tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t))
            cv2.imshow("AirStrings", overlay.draw(frame, snap))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if (cmd := key_to_command(key)) is not None:
                session.dispatch(cmd)
    finally:
        backend.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
