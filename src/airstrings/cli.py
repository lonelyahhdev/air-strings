"""Command line entry point: ``airstrings play|demo|parse``."""

from __future__ import annotations

import argparse
import sys

from airstrings.config import load_config
from airstrings.music.sequence import DEMO_SEQUENCES, parse_sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airstrings", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    play = sub.add_parser("play", help="start the webcam instrument")
    play.add_argument("--config", help="TOML file overriding defaults")
    play.add_argument("--instrument", help="guitar | piano")
    play.add_argument("--key", help="e.g. C, Am, F# minor")
    play.add_argument("--sequence", help='linear sequence, e.g. "C-D-D-A-G-E-B" (starts sequence mode)')
    play.add_argument("--demo-sequence", choices=sorted(DEMO_SEQUENCES), help="use a built-in sequence")
    play.add_argument("--camera", type=int, help="camera index")
    play.add_argument("--no-audio", action="store_true", help="skip the sound device")

    demo = sub.add_parser("demo", help="render a scripted performance to a WAV (no camera/sound card)")
    demo.add_argument("--out", default="airstrings_demo.wav")

    parse = sub.add_parser("parse", help="show how a sequence string is understood")
    parse.add_argument("text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "parse":
        try:
            seq = parse_sequence(args.text)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for i, step in enumerate(seq.steps, 1):
            print(f"{i:>2}. {step.kind.value:<5} {step.label:<6} midi={list(step.midi)}")
        return 0

    if args.cmd == "demo":
        from airstrings.demo import render_demo

        seconds = render_demo(args.out)
        print(f"wrote {args.out} ({seconds:.1f}s)")
        return 0

    config = load_config(args.config)
    if args.instrument:
        config.app.instrument = args.instrument
    if args.key:
        config.app.key = args.key
    if args.camera is not None:
        config.app.camera_index = args.camera
    sequence = args.sequence or (DEMO_SEQUENCES[args.demo_sequence] if args.demo_sequence else None)

    from airstrings.app import run

    try:
        run(config, sequence=sequence, audio=not args.no_audio)
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
