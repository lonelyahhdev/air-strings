import numpy as np
import pytest

from airstrings.audio.backend import OfflineBackend
from airstrings.audio.events import NoteOn, ReleaseAll
from airstrings.audio.mixer import Mixer
from airstrings.config import Config
from airstrings.demo import Segment, default_script, run_script
from airstrings.engine import commands as C
from airstrings.engine.session import Session
from airstrings.engine.state import Mode, PerformanceState
from airstrings.gestures.types import ChordIntent, Direction, Trigger
from airstrings.instruments import Instrument, PlayContext, create_instrument, instrument_names, register_instrument
from airstrings.music.sequence import StepKind, parse_sequence
from airstrings.music.theory import Chord, Key


def trig(direction=Direction.DOWN, intensity=0.7, openness=1.0, t=0.0):
    return Trigger(t, direction, intensity, 0.7, openness)


def notes(events):
    return [e for e in events if isinstance(e, NoteOn)]


def ctx(chord="C"):
    return PlayContext(Key.parse("C"), Chord.parse(chord))


# ---- instruments ------------------------------------------------------------------
def test_registry_has_builtin_instruments():
    assert instrument_names()[:2] == ["guitar", "piano"]
    with pytest.raises(KeyError, match="kazoo"):
        create_instrument("kazoo")


def test_guitar_downstroke_is_low_to_high_with_stagger():
    ev = notes(create_instrument("guitar").on_trigger(ctx("C"), trig()))
    assert [e.midi for e in ev] == [48, 52, 55, 60, 64, 67]
    assert [e.delay for e in ev] == sorted(e.delay for e in ev) and ev[0].delay == 0
    assert all(e.timbre == "guitar" for e in ev)


def test_guitar_upstroke_is_high_to_low_and_lighter():
    g = create_instrument("guitar")
    down, up = notes(g.on_trigger(ctx(), trig())), notes(g.on_trigger(ctx(), trig(Direction.UP)))
    assert [e.midi for e in up] == sorted((e.midi for e in up), reverse=True)
    assert len(up) == 4 and up[0].velocity < down[-1].velocity


def test_guitar_strum_damps_previous_and_palm_mute_shortens():
    g = create_instrument("guitar")
    events = g.on_trigger(ctx(), trig())
    assert isinstance(events[0], ReleaseAll) and events[0].timbre == "guitar"
    muted = notes(g.on_trigger(ctx(), trig(openness=0.1)))
    assert all(e.duration and e.duration < 0.5 for e in muted)
    assert all(e.duration is None for e in notes(events))


def test_harder_swing_means_louder_and_tighter_strum():
    g = create_instrument("guitar")
    soft, hard = notes(g.on_trigger(ctx(), trig(intensity=0.1))), notes(g.on_trigger(ctx(), trig(intensity=1.0)))
    assert hard[0].velocity > soft[0].velocity
    assert hard[-1].delay < soft[-1].delay


def test_piano_down_has_bass_and_block_chord_and_up_rolls():
    p = create_instrument("piano")
    down = notes(p.on_trigger(ctx("Am"), trig()))
    assert min(e.midi for e in down) < 48 and {e.midi % 12 for e in down} == {9, 0, 4}
    up = notes(p.on_trigger(ctx("Am"), trig(Direction.UP)))
    assert [e.midi for e in up] == sorted(e.midi for e in up)
    assert up[-1].delay > up[0].delay


def test_instruments_sequence_steps():
    seq = parse_sequence("C - Am")
    for name in ("guitar", "piano"):
        inst = create_instrument(name)
        note_step = parse_sequence("D").steps[0]
        assert [e.midi for e in notes(inst.on_step(ctx(), note_step, trig()))] == [62]
        assert inst.on_step(ctx(), seq.steps[1], trig()) == []  # rest
        chord_ev = notes(inst.on_step(ctx(), seq.steps[2], trig()))
        assert len(chord_ev) >= 3 and {e.midi % 12 for e in chord_ev} >= {9, 0, 4}


def test_custom_instrument_plugs_in_without_touching_core():
    @register_instrument
    class Bell(Instrument):
        name = "bell"
        timbre = "piano"

        def on_trigger(self, ctx, trigger):
            return [NoteOn(84, trigger.velocity, self.timbre)]

        def on_step(self, ctx, step, trigger):
            return [NoteOn(m + 24, trigger.velocity, self.timbre) for m in step.midi]

        def on_accompaniment(self, ctx, chord, trigger):
            return []

    state = PerformanceState(Key.parse("C"), "guitar")
    state.apply(C.SwitchInstrument("bell"))
    assert [e.midi for e in notes(state.on_trigger(trig()))] == [84]


# ---- state machine ----------------------------------------------------------------
def intent(degree, fingers=1, extended=False):
    return ChordIntent(fingers, degree, extended, True, True)


def test_switching_instrument_releases_old_voices_and_routes_to_new():
    s = PerformanceState(Key.parse("C"), "guitar")
    events = s.apply(C.SwitchInstrument("piano"))
    assert ReleaseAll("guitar", 0.1) in events
    assert all(e.timbre == "piano" for e in notes(s.on_trigger(trig())))
    assert s.apply(C.SwitchInstrument("piano")) == []  # no-op
    with pytest.raises(KeyError):
        s.apply(C.SwitchInstrument("kazoo"))
    assert s.instrument.name == "piano"  # failed switch changes nothing


def test_cycle_instrument_wraps():
    s = PerformanceState(Key.parse("C"), "guitar")
    names = instrument_names()
    for _ in range(len(names)):
        s.apply(C.CycleInstrument())
    assert s.instrument.name == "guitar"


def test_free_play_follows_chord_intent_and_fist_silences():
    s = PerformanceState(Key.parse("C"), "guitar")
    s.apply_chord_intent(intent(5))
    assert s.chord.name == "G"
    assert {e.midi % 12 for e in notes(s.on_trigger(trig()))} == {7, 11, 2}
    s.apply_chord_intent(intent(5, extended=True))
    assert s.chord.name == "G7"
    cut = s.apply_chord_intent(intent(None, fingers=0))
    assert cut and isinstance(cut[0], ReleaseAll)
    assert s.on_trigger(trig()) == []


def test_changed_false_is_ignored():
    s = PerformanceState(Key.parse("C"), "guitar")
    assert s.apply_chord_intent(ChordIntent(3, 5, False, True, False)) == []
    assert s.chord.name == "C"


def test_key_change_retargets_chords():
    s = PerformanceState(Key.parse("C"), "guitar")
    s.apply_chord_intent(intent(4))
    assert s.chord.name == "F"
    s.apply(C.TransposeKey(2))
    assert s.chord.name == "G"  # IV of D
    s.apply(C.SetKey(Key.parse("A minor")))
    assert s.chord.name == "Dm"
    s.apply(C.ToggleKeyMode())
    assert s.key.name == "A major"


def test_sequence_mode_advances_cursor_and_auto_harmonises():
    s = PerformanceState(Key.parse("C"), "guitar")
    s.apply(C.LoadSequence("C-D-D-A-G-E-B"))
    assert s.mode is Mode.SEQUENCE
    played = []
    chords = []
    for _ in range(7):
        ev = s.on_trigger(trig())
        main = [e for e in notes(ev) if e.duration is None]
        played.append(main[-1].midi)
        chords.append(s.accompaniment.name)
    assert played == [60, 62, 62, 69, 67, 64, 71]
    assert chords == ["C", "G", "G", "F", "C", "C", "G"]
    assert s.cursor.index == 0  # looped


def test_sequence_first_trigger_includes_accompaniment_then_melody():
    s = PerformanceState(Key.parse("C"), "piano")
    s.apply(C.LoadSequence("C D"))
    ev = notes(s.on_trigger(trig()))
    assert any(e.duration for e in ev) and ev[-1].midi == 60 and ev[-1].duration is None


def test_sequence_ignores_chord_hand_and_rests_do_nothing():
    s = PerformanceState(Key.parse("C"), "guitar")
    s.apply(C.LoadSequence("C - D"))
    assert s.apply_chord_intent(intent(5)) == []
    assert s.on_trigger(trig())
    assert s.on_trigger(trig()) == []
    assert s.cursor.index == 2


def test_sequence_chord_steps_and_reset_and_mode_toggle():
    s = PerformanceState(Key.parse("C"), "guitar")
    s.apply(C.LoadSequence("C G Am F"))
    s.on_trigger(trig())
    assert s.accompaniment.name == "C"
    s.on_trigger(trig())
    assert s.accompaniment.name == "G"
    s.apply(C.ResetSequence())
    assert s.cursor.index == 0
    s.apply(C.ToggleMode())
    assert s.mode is Mode.FREE
    assert s.sounding_chord.name == "C"


def test_non_looping_sequence_ends():
    s = PerformanceState(Key.parse("C"), "guitar")
    s.apply(C.LoadSequence(parse_sequence("C D", loop=False)))
    assert s.on_trigger(trig()) and s.on_trigger(trig())
    assert s.on_trigger(trig()) == []


def test_bad_sequence_leaves_state_untouched():
    s = PerformanceState(Key.parse("C"), "guitar")
    with pytest.raises(ValueError):
        s.apply(C.LoadSequence("C Q"))
    assert s.mode is Mode.FREE


def test_listeners_hear_state_changes():
    s = PerformanceState(Key.parse("C"), "guitar")
    heard = []
    s.subscribe(heard.append)
    s.apply(C.SwitchInstrument("piano"))
    s.apply(C.ToggleMode())
    assert heard == ["instrument: piano", "mode: sequence"]


def test_step_kind_enum_is_exposed():
    assert StepKind.NOTE.value == "note"


# ---- full pipeline ----------------------------------------------------------------
def make_session():
    cfg = Config()
    mixer = Mixer(cfg.audio.sample_rate)
    backend = OfflineBackend(mixer)
    return Session(cfg, backend.submit), backend


def test_end_to_end_scripted_performance():
    session, backend = make_session()
    snaps = run_script(session, backend, default_script())
    audio = backend.audio()
    assert len(audio) == pytest.approx(27 * 44100, abs=2000)
    assert np.abs(audio).max() < 1.0 and np.abs(audio).max() > 0.2  # audible, not clipping
    assert not np.isnan(audio).any()

    chords_by_instrument = {(s.instrument, s.mode, s.chord_name) for s in snaps}
    assert {("guitar", "free", n) for n in ("C", "F", "G", "Am", "G7")} <= chords_by_instrument
    assert {("piano", "free", n) for n in ("C", "F", "G", "Am")} <= chords_by_instrument
    assert any(s.mode == "sequence" for s in snaps)
    assert snaps[-1].sequence_upcoming  # ribbon has something to show

    # loudness differs while strumming vs. after the final rest (it rings, then decays)
    sr = 44100
    rms = lambda a, b: float(np.sqrt((audio[int(a * sr) : int(b * sr)] ** 2).mean()))  # noqa: E731
    assert rms(0.5, 1.9) > 0.02 and rms(0.0, 0.2) < rms(0.5, 1.9)


def test_gesture_driven_instrument_swap():
    session, backend = make_session()
    fists = [Segment(0.5, 0, strum_hz=0.0), Segment(1.5, 0, strum_hz=0.0)]
    # both fists: the demo's rhythm hand is open, so drive the swap with explicit frames
    from airstrings.tracking.synthetic import fingers, make_hand
    from airstrings.tracking.types import Side, TrackingFrame

    t = 0.0
    for _ in range(45):
        hands = (make_hand(Side.LEFT, fingers(0), (0.25, 0.6)), make_hand(Side.RIGHT, fingers(0), (0.75, 0.7)))
        snap = session.process(TrackingFrame(t, hands))
        t += 1 / 30
    assert snap.instrument == "piano"
    assert snap.switch_progress == 1.0
    assert fists  # (segments above only document the scenario)


def test_snapshot_ladder_reflects_key_and_seventh():
    session, _ = make_session()
    from airstrings.tracking.synthetic import fingers, make_hand
    from airstrings.tracking.types import Side, TrackingFrame

    frame = TrackingFrame(0.0, (make_hand(Side.LEFT, fingers(1), (0.25, 0.6)),))
    assert [n for _, n in session.process(frame).chord_ladder] == ["C", "F", "G", "Am", "Dm"]
    session.dispatch(C.SetKey(Key.parse("G")))
    assert [n for _, n in session.process(frame).chord_ladder][:3] == ["G", "C", "D"]


def test_strum_line_nudge_is_clamped():
    session, _ = make_session()
    for _ in range(100):
        session.dispatch(C.NudgeStrumLine(0.05))
    assert session.strum.line_y == pytest.approx(0.9)
    for _ in range(100):
        session.dispatch(C.NudgeStrumLine(-0.05))
    assert session.strum.line_y == pytest.approx(0.15)


def test_harmony_side_can_be_swapped_for_lefties():
    cfg = Config()
    cfg.gestures.harmony_side = "right"
    session = Session(cfg, lambda events: None)
    assert session.harmony_side.value == "right" and session.rhythm_side.value == "left"


def test_status_message_appears_then_expires():
    from airstrings.engine.session import STATUS_SECONDS
    from airstrings.tracking.types import TrackingFrame

    session, _ = make_session()
    session.dispatch(C.SwitchInstrument("piano"))
    assert session.process(TrackingFrame(10.0)).status == "instrument: piano"
    assert session.process(TrackingFrame(10.0 + STATUS_SECONDS - 0.1)).status == "instrument: piano"
    assert session.process(TrackingFrame(10.0 + STATUS_SECONDS + 0.1)).status == ""
