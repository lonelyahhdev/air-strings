import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from airstrings import cli  # noqa: E402
from airstrings.app import key_to_command  # noqa: E402
from airstrings.config import Config, load_config  # noqa: E402
from airstrings.demo import Segment, run_script  # noqa: E402
from airstrings.engine import commands as C  # noqa: E402
from airstrings.engine.session import Session  # noqa: E402
from airstrings.audio.backend import OfflineBackend  # noqa: E402
from airstrings.audio.mixer import Mixer  # noqa: E402
from airstrings.ui.overlay import Overlay  # noqa: E402


def test_overlay_draws_in_every_state():
    cfg = Config()
    backend = OfflineBackend(Mixer())
    session = Session(cfg, backend.submit)
    overlay = Overlay()
    script = [
        Segment(1.0, 2),
        Segment(1.0, 3, high=True, command=C.SwitchInstrument("piano")),
        Segment(1.0, 1, command=C.LoadSequence("C D E")),
    ]
    snaps = run_script(session, backend, script)
    for snap in (snaps[5], snaps[40], snaps[-1]):
        frame = np.zeros((720, 1280, 3), np.uint8)
        out = overlay.draw(frame, snap)
        assert out is frame and out.any()
    for size in ((480, 640), (360, 320)):  # small windows must not crash
        overlay.draw(np.zeros((*size, 3), np.uint8), snaps[40])


def test_overlay_strum_flash_changes_pixels():
    cfg = Config()
    backend = OfflineBackend(Mixer())
    session = Session(cfg, backend.submit)
    snaps = run_script(session, backend, [Segment(1.5, 1)])
    flashed = next(s for s in snaps if s.last_trigger and s.t - s.last_trigger.t < 0.05)
    quiet = snaps[0]
    a, b = (Overlay().draw(np.zeros((720, 1280, 3), np.uint8), s) for s in (flashed, quiet))
    assert (a != b).any()


def test_key_to_command_mapping():
    assert key_to_command(ord("1")) == C.SwitchInstrument("guitar")
    assert key_to_command(ord("2")) == C.SwitchInstrument("piano")
    assert key_to_command(ord("9")) is None
    assert key_to_command(ord("m")) == C.ToggleMode()
    assert key_to_command(ord(".")) == C.TransposeKey(1)
    assert key_to_command(255) is None and key_to_command(-1) is None


def test_load_config_overrides_and_validation(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        '[gestures]\nharmony_side = "right"\nstrum_line_y = 0.6\n[gestures.chord_map]\n1 = 1\n2 = 5\n'
        '[app]\nkey = "Am"\ninstrument = "piano"\n'
    )
    cfg = load_config(path)
    assert cfg.gestures.harmony_side == "right" and cfg.rhythm_side == "left"
    assert cfg.gestures.chord_map == {1: 1, 2: 5}
    assert cfg.app.key == "Am"
    path.write_text("[gestures]\nbogus = 1\n")
    with pytest.raises(ValueError, match="bogus"):
        load_config(path)
    path.write_text("[nope]\nx = 1\n")
    with pytest.raises(ValueError, match="nope"):
        load_config(path)
    path.write_text('[gestures]\nharmony_side = "middle"\n')
    with pytest.raises(ValueError, match="harmony_side"):
        load_config(path)


def test_shipped_default_config_matches_code_defaults():
    import pathlib

    shipped = load_config(pathlib.Path(__file__).parent.parent / "config" / "default.toml")
    assert shipped == Config()


def test_cli_parse_and_demo(tmp_path, capsys):
    assert cli.main(["parse", "C-D-D-A-G-E-B"]) == 0
    out = capsys.readouterr().out
    assert "midi=[60]" in out and out.count("note") == 7
    assert cli.main(["parse", "C Q"]) == 2
    wav = tmp_path / "d.wav"
    assert cli.main(["demo", "--out", str(wav)]) == 0
    assert wav.stat().st_size > 100_000
