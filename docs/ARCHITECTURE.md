# Architecture

## Idea: an abstract workspace, not a fake instrument

There are no virtual frets or key positions to hit. The two hands have
different jobs, and both are meant to be learnable in seconds:

| Hand | Job | Gestures |
|---|---|---|
| Chord hand | *What* harmony | finger count → chord (I IV V vi ii by default), hand height → adds a 7th, fist → silence |
| Playing hand | *When* and *how hard* | palm swing through a horizontal line → hit; swing direction → down/up; swing speed → velocity; hand openness → ringing vs. damped |

The strum line and chord ladder are drawn on screen, so users are guided
without reading anything. Dynamics use speed measured in *hand-lengths per
second*, so they don't change with how far the player stands from the camera.

## Pipeline

```
 camera ─► MediaPipeHandTracker ─► TrackingFrame ─► Session.process ─► Snapshot ─► Overlay (OpenCV)
              (only place that        (Hand: 21        │   ▲
               knows MediaPipe)        landmarks)      │   └──── Commands (keyboard / both-fists gesture)
                                                       ▼
                         ChordSelector ─► ChordIntent ─┐
                         StrumDetector ─► Trigger ─────┼─► PerformanceState ─► Instrument ─► AudioEvents ─► Mixer ─► sound card
                         SwitchDetector ─► Command ────┘    (state machine)     (Guitar/Piano)   NoteOn/ReleaseAll
```

Each arrow is a plain dataclass, so every stage is testable on its own, and
the whole pipeline runs headless on synthetic hands (`airstrings demo`).

## Layout

```
src/airstrings/
  tracking/     types (Hand, TrackingFrame), geometric features, One-Euro smoothing,
                MediaPipe adapter, synthetic hands for tests/demos
  gestures/     ChordSelector, StrumDetector, SwitchDetector; gesture vocabulary (Trigger, ChordIntent)
  music/        theory (Key, Chord, voicings), sequence parser + cursor, melody Harmonizer
  instruments/  Instrument ABC + registry; guitar.py, piano.py
  engine/       commands, PerformanceState (the state machine), Session (wires it all)
  audio/        events, synth (Karplus-Strong guitar, additive piano), Mixer, backends
  ui/           Overlay
  app.py        realtime loop;  cli.py  entry point;  demo.py  scripted performance;  config.py
tests/          unit tests per layer + end-to-end scripted performance
```

Dependency direction is strictly downward: `tracking` knows nothing about
music, `instruments` know nothing about cameras or sound devices, `audio`
knows nothing about instruments.

## State machine

`PerformanceState` holds *instrument × mode × key × chord × sequence cursor*.
Inputs are `ChordIntent`, `Trigger` and `Command`s; outputs are audio events.

* **Free play**: the chord hand picks the chord; each trigger asks the active instrument to play it.
* **Sequence**: each trigger consumes the next `SequenceStep`. A `Harmonizer` chooses a diatonic chord containing each melody note (sticky, to avoid chord changes on every note) and the instrument adds quiet backing when the chord changes.
* Swapping instruments silences the old instrument's voices and routes everything to the new one. Instruments are created once and cached; a failed switch changes nothing.
* Listeners can subscribe to state changes (the UI uses this for its status line).

## Library choices

| Need | Choice | Why |
|---|---|---|
| Hand tracking | **MediaPipe Hand Landmarker** (Tasks API) | Two hands, 21 landmarks each, real-time on CPU, handedness included, no training. Alternatives (OpenPose, YOLO-pose) are heavier or lack finger landmarks. Wrapped behind `TrackingFrame`, so it can be swapped. |
| Camera, overlay | **OpenCV** | Capture, drawing and a window in one dependency. A Qt/Pygame UI can replace `ui/` later because it only reads a `Snapshot`. |
| Audio out | **sounddevice** (PortAudio) + **NumPy** | Callback API with low latency and sample-accurate scheduling; works on all three platforms. No sample libraries or soundfonts to ship. |
| Synthesis | **Karplus-Strong** guitar, **additive** piano | Both are plucked/struck strings. Each note is rendered once into a buffer (vectorised NumPy, ~5-60 ms) and cached, so the audio callback only sums slices: cheap and glitch-free. Strum staggering is sample-accurate. |
| Alternatives considered | FluidSynth/soundfonts, pyo, pygame.mixer | Soundfonts sound richer but add a native dependency and large assets; pygame.mixer has no scheduling; pyo is harder to install. A soundfont timbre can be added later through the timbre registry. |

Python ≥ 3.11 (uses `tomllib`, `match`). MediaPipe's Tasks API is required
(its legacy `solutions` API is gone in recent versions).

## Extending

**New instrument** (e.g. ukulele, bass, drums): subclass `Instrument` in
`instruments/`, implement `on_trigger`, `on_step`, `on_accompaniment`, decorate
with `@register_instrument`, and import it in `instruments/__init__.py`. It
appears in the UI tabs, number-key shortcuts and the both-fists cycle
automatically. If it needs a new sound, add a renderer with
`@register_timbre("name")` in `audio/synth.py`.

**New gesture**: add a detector under `gestures/` that maps a `Hand` to an intent or `Command`, call it from `Session.process`.

**New tracker**: produce a `TrackingFrame` of `Hand`s (normalised, mirrored image coordinates, y down).

## Known limits and next steps

* Timing resolution is one camera frame (~33 ms at 30 fps). `Trigger.t` is where the hit happened; a later improvement is to schedule the audio at `t` using the measured crossing time, and to run capture + tracking on their own thread so inference never delays the UI.
* Finger counting uses a simple "tip farther than PIP from wrist" test. It works for palm-forward hands; sideways or strongly tilted hands will be less reliable. A small learned classifier or per-user calibration would improve it.
* The first note of a not-yet-cached pitch is rendered on demand (up to ~60 ms for low piano notes). Sample caches are pre-warmed in the background at startup, so this only affects the first moments.
* Mono output; guitar and piano are synthetic. A soundfont/sample backend would be the next step for realism.
* Verified headless (synthetic hands, offline mixer, stubbed MediaPipe adapter) and the real MediaPipe model loads and runs. A live webcam session and a real sound device have not been exercised in CI, so gesture thresholds in `config/default.toml` will likely need tuning on real hands.
