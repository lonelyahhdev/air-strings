import numpy as np
import pytest

from airstrings.audio.backend import OfflineBackend
from airstrings.audio.events import NoteOn, ReleaseAll
from airstrings.audio.mixer import Mixer
from airstrings.audio.synth import SampleBank, render_guitar, render_piano, timbre_names
from airstrings.music.theory import midi_to_hz

SR = 44100


def fundamental(y, midi, sr=SR):
    seg = y[int(0.05 * sr) : int(0.45 * sr)]
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, "full")[len(seg) - 1 :]
    period = sr / midi_to_hz(midi)
    lo, hi = int(period * 0.7), int(period * 1.4)
    lag = lo + int(np.argmax(ac[lo:hi]))
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    lag += 0.5 * (a - c) / (a - 2 * b + c)
    return sr / lag


@pytest.mark.parametrize("midi", [40, 52, 64, 76])
def test_guitar_is_in_tune(midi):
    cents = 1200 * np.log2(fundamental(render_guitar(midi, 90, SR), midi) / midi_to_hz(midi))
    assert abs(cents) < 5


@pytest.mark.parametrize("midi", [40, 60, 84])
def test_piano_is_close_to_pitch(midi):
    cents = 1200 * np.log2(fundamental(render_piano(midi, 90, SR), midi) / midi_to_hz(midi))
    assert abs(cents) < 15  # inharmonic stretch is intentional


def test_synth_outputs_are_normalised_and_decay():
    for fn in (render_guitar, render_piano):
        y = fn(60, 100, SR)
        assert y.dtype == np.float32 and np.abs(y).max() == pytest.approx(1.0)
        assert np.abs(y[: SR // 4]).mean() > 3 * np.abs(y[-SR // 4 :]).mean()


def test_louder_velocity_is_brighter_for_guitar():
    soft, hard = render_guitar(60, 20, SR), render_guitar(60, 120, SR)

    def centroid(y):
        mag = np.abs(np.fft.rfft(y[:8192]))
        return float((mag * np.arange(len(mag))).sum() / mag.sum())

    assert centroid(hard) > centroid(soft)


def test_sample_bank_caches_and_validates():
    bank = SampleBank(SR)
    assert bank.get("guitar", 60, 90) is bank.get("guitar", 60, 95)  # same brightness bucket
    assert set(timbre_names()) >= {"guitar", "piano"}
    with pytest.raises(KeyError):
        bank.get("kazoo", 60, 90)


def render(mixer, seconds):
    return np.concatenate([mixer.render(441) for _ in range(int(seconds * 100))])


def test_mixer_silence_then_note():
    m = Mixer(SR)
    assert not render(m, 0.1).any()
    m.submit([NoteOn(60, 100, "piano")])
    out = render(m, 0.2)
    assert np.abs(out).max() > 0.05


def test_mixer_schedules_notes_sample_accurately():
    m = Mixer(SR)
    m.submit([NoteOn(60, 100, "guitar", delay=0.05)])
    out = render(m, 0.2)
    first = int(np.argmax(np.abs(out) > 1e-4))
    assert first == pytest.approx(0.05 * SR, abs=2)


def test_mixer_polyphony_sums_and_voices_expire():
    m = Mixer(SR)
    m.submit([NoteOn(n, 100, "guitar") for n in (48, 52, 55, 60, 64, 67)])
    render(m, 0.05)
    assert m.active_voices == 6
    chord = render(m, 0.1)
    m2 = Mixer(SR)
    m2.submit([NoteOn(48, 100, "guitar")])
    render(m2, 0.05)
    single = render(m2, 0.1)
    assert np.sqrt((chord**2).mean()) > 1.5 * np.sqrt((single**2).mean())
    render(m, 4.0)
    assert m.active_voices == 0


def test_mixer_release_all_fades_out():
    m = Mixer(SR)
    m.submit([NoteOn(60, 100, "piano")])
    render(m, 0.1)
    m.submit([ReleaseAll("piano", fade=0.05)])
    render(m, 0.1)
    assert m.active_voices == 0
    assert not render(m, 0.1).any()


def test_release_all_only_hits_matching_timbre():
    m = Mixer(SR)
    m.submit([NoteOn(60, 100, "piano"), NoteOn(60, 100, "guitar")])
    render(m, 0.05)
    m.submit([ReleaseAll("piano", fade=0.01)])
    render(m, 0.05)
    assert m.active_voices == 1


def test_release_cancels_not_yet_started_notes():
    m = Mixer(SR)
    m.submit([NoteOn(60, 100, "guitar", delay=0.5), ReleaseAll(None, 0.01)])
    assert not render(m, 0.8).any()


def test_duration_releases_note():
    m = Mixer(SR)
    m.submit([NoteOn(60, 100, "piano", duration=0.1, release=0.05)])
    out = render(m, 0.5)
    assert np.abs(out[int(0.25 * SR) :]).max() < 1e-6


def test_restrike_same_pitch_chokes_previous_voice():
    m = Mixer(SR)
    m.submit([NoteOn(60, 100, "guitar")])
    render(m, 0.05)
    m.submit([NoteOn(60, 100, "guitar")])
    render(m, 0.2)
    assert m.active_voices == 1


def test_voice_stealing_caps_polyphony():
    m = Mixer(SR, max_voices=8)
    m.submit([NoteOn(40 + i, 90, "guitar") for i in range(20)])
    render(m, 0.2)
    assert m.active_voices <= 8


def test_offline_backend_wav_roundtrip(tmp_path):
    import wave

    backend = OfflineBackend(Mixer(SR))
    backend.submit([NoteOn(64, 100, "piano")])
    backend.advance(0.5)
    path = tmp_path / "x.wav"
    backend.write_wav(str(path))
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (SR, 1, SR // 2)
