"""Turn a lone melody note into a sensible backing chord."""

from __future__ import annotations

from airstrings.music.theory import Chord, Key

# Degrees tried in this order: tonic, dominant, subdominant, then the rest.
_PRIORITY = (1, 5, 4, 6, 2, 3, 7)


class Harmonizer:
    """Picks a diatonic chord containing the melody note.

    Prefers staying on the current chord (so accompaniment doesn't change on
    every note), otherwise the highest-priority diatonic chord that fits.
    """

    def __init__(self, key: Key):
        self.key = key
        self._chords = {d: key.diatonic_chord(d) for d in _PRIORITY}

    def chord_for_note(self, midi: int, current: Chord | None = None) -> Chord:
        pc = midi % 12
        if current is not None and current.contains_pc(pc) and current in self._chords.values():
            return current
        for degree in _PRIORITY:
            chord = self._chords[degree]
            if chord.contains_pc(pc):
                return chord
        return self._chords[1]  # chromatic note: fall back to tonic
