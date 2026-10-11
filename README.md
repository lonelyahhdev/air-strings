# AirStrings

Play guitar and piano in the air. Your webcam watches your hands; one hand
chooses the harmony, the other plays the rhythm. No musical training needed.

```
   CHORD HAND (left)                    PLAYING HAND (right): pinch, don't wave
   ┌─────────────────┐                  thumb + index   = play the chord
   │ 1 finger  C     │                  thumb + middle  = bass note
   │ 2 fingers F     │                  thumb + ring    = rolling arpeggio
   │ 3 fingers G     │                  tap = a hit, hold = sustain, let go = stop
   │ 4 fingers Am    │                  quicker pinch = louder
   │ 5 fingers Dm    │
   └─────────────────┘   fist = silence      raise the hand = add a 7th
                         both fists for 1 s = swap instrument
```

Only your fingers move: rest the playing hand wherever it is comfortable and
tap. The window labels the three fingertips and lights up the one you are
about to pinch.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
# Linux only, for audio output:  sudo apt install libportaudio2

airstrings play                                   # free play, guitar, key of C
airstrings play --instrument piano --key Am
airstrings play --sequence "C-D-D-A-G-E-B"        # follow-along: one pinch = next note
airstrings play --sequence "0 2 3 5 3 2 0 -"      # guitar fret numbers
airstrings play --sequence "C G Am F"             # chord progression
airstrings play --demo-sequence twinkle
airstrings play --no-audio                        # camera + overlay only
```

The hand-tracking model (`hand_landmarker.task`, ~8 MB) is downloaded to
`~/.cache/airstrings` the first time you run it.

Without a camera or sound card:

```bash
airstrings demo --out demo.wav        # scripted performance rendered to a WAV
airstrings parse "C-D-D-A-G-E-B"      # check how a sequence is understood
pytest                                # 119 tests, all headless
```

Keys in the window: `1`/`2` instrument, `Tab` next, `m` free/sequence mode,
`r` restart sequence, `,` `.` transpose, `n` major/minor, `]` / `[` make pinching
easier / harder to trigger, `q` quit. Options live in [`config/default.toml`](config/default.toml).

## Sequence notation

| Style | Example | Meaning |
|---|---|---|
| Note names | `C D D A G E B`, `C-D-D-A-G-E-B`, `C5`, `F#3`, `Bb` | one note per pinch |
| Fret numbers | `0 2 3 5`, `5:3` | fret on the default string, or `string:fret` |
| Chords | `Am F C G` | one strum per pinch |
| Rests | `-` `.` `_` | skip a beat |

In sequence mode the chord hand is not needed: a small harmonizer picks a
diatonic chord under each melody note and plays soft backing, so a bare
melody becomes a full arrangement.

## More

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): layout, library choices, data flow, and how to add an instrument.
