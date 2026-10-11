import pytest

from airstrings.music.harmony import Harmonizer
from airstrings.music.sequence import SequenceCursor, StepKind, fret_to_midi, parse_sequence
from airstrings.music.theory import Chord, Key, midi_to_name, parse_note


def test_parse_note():
    assert parse_note("C4") == 60
    assert parse_note("c") == 60
    assert parse_note("A") == 69
    assert parse_note("Bb3") == 58
    assert parse_note("F#5") == 78
    with pytest.raises(ValueError):
        parse_note("H")


def test_key_diatonic_chords_c_major():
    key = Key.parse("C")
    names = [key.diatonic_chord(d).name for d in range(1, 8)]
    assert names == ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    assert key.diatonic_chord(5, seventh=True).name == "G7"
    assert key.diatonic_chord(1, seventh=True).name == "Cmaj7"
    assert key.diatonic_chord(6, seventh=True).name == "Am7"


def test_key_minor_and_spelling():
    key = Key.parse("A minor")
    assert [key.diatonic_chord(d).name for d in (1, 4, 5)] == ["Am", "Dm", "Em"]
    assert key.diatonic_chord(3).name == "C"
    assert Key.parse("F").diatonic_chord(4).name == "Bb"  # flat keys spell with flats
    assert Key.parse("Bbm").mode == "minor"


def test_chord_voicings():
    c = Chord.parse("C")
    assert c.stacked(40, 6) == (48, 52, 55, 60, 64, 67)
    assert c.bass_note(36) == 36
    voicing = c.close_voicing(64)
    assert len(voicing) == 3
    assert {n % 12 for n in voicing} == {0, 4, 7}
    assert list(voicing) == sorted(voicing)
    assert max(voicing) - min(voicing) < 12


def test_parse_user_example_sequence():
    seq = parse_sequence("C-D-D-A-G-E-B")
    assert [s.label for s in seq.steps] == list("CDDAGEB")
    assert all(s.kind is StepKind.NOTE for s in seq.steps)
    assert [midi_to_name(s.midi[0]) for s in seq.steps] == ["C4", "D4", "D4", "A4", "G4", "E4", "B4"]


def test_parse_spaces_and_rests_and_octaves():
    seq = parse_sequence("C D - E5 .")
    assert [s.kind for s in seq.steps] == [StepKind.NOTE, StepKind.NOTE, StepKind.REST, StepKind.NOTE, StepKind.REST]
    assert seq.steps[3].midi == (76,)


def test_parse_frets():
    seq = parse_sequence("0 2 3 6:0 1:3", default_string=4)
    assert [s.midi[0] for s in seq.steps] == [50, 52, 53, 40, 67]
    assert fret_to_midi(1, 0) == 64


def test_parse_chords():
    seq = parse_sequence("C G Am F")  # Am makes bare letters chords
    assert all(s.kind is StepKind.CHORD for s in seq.steps)
    assert [s.label for s in seq.steps] == ["C", "G", "Am", "F"]


def test_parse_errors():
    with pytest.raises(ValueError, match="H"):
        parse_sequence("C H")
    with pytest.raises(ValueError):
        parse_sequence("   ")


def test_cursor_loops_and_finishes():
    cur = SequenceCursor(parse_sequence("C D E"))
    assert [cur.next_step().label for _ in range(4)] == ["C", "D", "E", "C"]
    assert cur.upcoming(4) == ["D", "E", "C", "D"]
    once = SequenceCursor(parse_sequence("C D", loop=False))
    once.next_step(), once.next_step()
    assert once.next_step() is None and once.upcoming(3) == []
    once.reset()
    assert once.next_step().label == "C"


def test_harmonizer_picks_fitting_and_sticky_chords():
    h = Harmonizer(Key.parse("C"))
    assert h.chord_for_note(60).name == "C"
    assert h.chord_for_note(62).name == "G"  # D -> V
    assert h.chord_for_note(71, None).name == "G"  # B -> V
    assert h.chord_for_note(69, None).name == "F"  # A -> IV
    # stays on the current chord while the note fits it
    assert h.chord_for_note(64, Chord.parse("Am")).name == "Am"
    assert h.chord_for_note(66).name == "C"  # chromatic -> tonic
