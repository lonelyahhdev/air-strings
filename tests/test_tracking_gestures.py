
import pytest

from airstrings.config import GestureConfig
from airstrings.gestures.chord_selector import ChordSelector
from airstrings.gestures.strum_detector import StrumDetector
from airstrings.gestures.switch_detector import SwitchDetector
from airstrings.gestures.types import Direction
from airstrings.tracking.features import count_fingers, finger_states, hand_scale, openness, palm_center
from airstrings.tracking.smoothing import OneEuroFilter
from airstrings.tracking.synthetic import fingers, make_hand
from airstrings.tracking.types import Side


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("n", range(6))
def test_finger_counting(side, n):
    assert count_fingers(make_hand(side, fingers(n))) == n


def test_finger_state_identity():
    hand = make_hand(Side.LEFT, (False, True, False, False, True))
    assert finger_states(hand) == (False, True, False, False, True)


def test_openness_and_scale_and_palm():
    assert openness(make_hand(Side.RIGHT, fingers(0))) < 0.1
    assert openness(make_hand(Side.RIGHT, fingers(5))) > 0.8
    assert hand_scale(make_hand(Side.RIGHT, length=0.1)) == pytest.approx(0.1, rel=0.05)
    x, y = palm_center(make_hand(Side.RIGHT, center=(0.3, 0.6)))
    assert (x, y) == pytest.approx((0.3, 0.6), abs=0.005)


def test_one_euro_tracks_step_and_smooths_noise():
    f = OneEuroFilter()
    out = [f(1.0, i / 30) for i in range(60)]
    assert out[-1] == pytest.approx(1.0, abs=1e-3)
    f.reset()
    assert f(5.0, 0.0) == 5.0


# ---- chord selector ---------------------------------------------------------------
def feed(selector, hand, t0, seconds, fps=30):
    last = None
    for i in range(int(seconds * fps)):
        last = selector.update(hand, t0 + i / fps)
    return last


def test_chord_selector_maps_fingers_after_settling():
    sel = ChordSelector(GestureConfig())
    intent = sel.update(make_hand(Side.LEFT, fingers(2), (0.25, 0.6)), 0.0)
    assert intent.degree is None or not intent.changed  # not settled yet
    intent = feed(sel, make_hand(Side.LEFT, fingers(2), (0.25, 0.6)), 0.0, 0.3)
    assert (intent.fingers, intent.degree, intent.extended) == (2, 4, False)


def test_chord_selector_ignores_flicker():
    sel = ChordSelector(GestureConfig())
    feed(sel, make_hand(Side.LEFT, fingers(3), (0.25, 0.6)), 0.0, 0.3)
    # a single stray frame of a different count must not change the chord
    sel.update(make_hand(Side.LEFT, fingers(1), (0.25, 0.6)), 0.31)
    intent = sel.update(make_hand(Side.LEFT, fingers(3), (0.25, 0.6)), 0.34)
    assert intent.degree == 5 and not intent.changed


def test_chord_selector_fist_is_silence_and_hand_loss_keeps_chord():
    sel = ChordSelector(GestureConfig())
    feed(sel, make_hand(Side.LEFT, fingers(2), (0.25, 0.6)), 0.0, 0.3)
    fist = feed(sel, make_hand(Side.LEFT, fingers(0), (0.25, 0.6)), 0.3, 0.3)
    assert fist.degree is None and fist.fingers == 0
    feed(sel, make_hand(Side.LEFT, fingers(4), (0.25, 0.6)), 0.6, 0.3)
    gone = sel.update(None, 1.0)
    assert gone.degree == 6 and not gone.present and not gone.changed


def test_chord_selector_seventh_zone_with_hysteresis():
    cfg = GestureConfig()
    sel = ChordSelector(cfg)
    up = feed(sel, make_hand(Side.LEFT, fingers(3), (0.25, cfg.seventh_zone_y - 0.1)), 0.0, 0.3)
    assert up.extended
    still = sel.update(make_hand(Side.LEFT, fingers(3), (0.25, cfg.seventh_zone_y + 0.03)), 0.4)
    assert still.extended  # inside hysteresis band
    down = sel.update(make_hand(Side.LEFT, fingers(3), (0.25, cfg.seventh_zone_y + 0.2)), 0.45)
    assert not down.extended and down.changed


# ---- strum detector ---------------------------------------------------------------
def swing(det, ys, fps=30, t0=0.0):
    out = []
    for i, y in enumerate(ys):
        out += det.update(make_hand(Side.RIGHT, fingers(5), (0.7, y)), t0 + i / fps)
    return out


def test_strum_down_then_up():
    cfg = GestureConfig()
    det = StrumDetector(cfg)
    line = cfg.strum_line_y
    down = [line - 0.2, line - 0.12, line - 0.04, line + 0.04, line + 0.12, line + 0.2]
    trig = swing(det, down + down[::-1])
    assert [t.direction for t in trig] == [Direction.DOWN, Direction.UP]
    assert 0 < trig[0].intensity <= 1 and trig[0].openness > 0.8


def test_strum_speed_scales_intensity():
    cfg = GestureConfig()
    line = cfg.strum_line_y
    slow = swing(StrumDetector(cfg), [line - 0.03, line - 0.01, line + 0.02, line + 0.04, line + 0.06], fps=10)
    fast = swing(StrumDetector(cfg), [line - 0.25, line - 0.1, line + 0.1, line + 0.25])
    assert fast and (not slow or fast[0].intensity > slow[0].intensity)
    assert fast[0].velocity > 90


def test_strum_ignores_slow_drift_and_jitter():
    cfg = GestureConfig()
    line = cfg.strum_line_y
    drift = [line - 0.02 + 0.001 * i for i in range(40)]  # crosses the line very slowly
    assert swing(StrumDetector(cfg), drift) == []
    jitter = [line + 0.002 * (-1) ** i for i in range(40)]
    assert swing(StrumDetector(cfg), jitter) == []


def test_strum_resets_when_hand_disappears():
    cfg = GestureConfig()
    det = StrumDetector(cfg)
    line = cfg.strum_line_y
    swing(det, [line - 0.2])
    assert det.update(None, 0.1) == []
    # reappearing below the line is not a crossing
    assert swing(det, [line + 0.2], t0=0.2) == []


def test_strum_line_is_adjustable():
    cfg = GestureConfig()
    det = StrumDetector(cfg)
    det.line_y = 0.3
    ys = [0.15, 0.25, 0.35, 0.45]
    assert len(swing(det, ys)) == 1


def test_strum_amplitude_follows_hand_size_not_camera_distance():
    cfg = GestureConfig()
    line = cfg.strum_line_y

    def go(length):
        det = StrumDetector(cfg)
        out = []
        for i, y in enumerate([line - 0.1, line - 0.03, line + 0.03, line + 0.1]):
            out += det.update(make_hand(Side.RIGHT, fingers(5), (0.7, y), length=length), i / 30)
        return out

    near, far = go(0.2), go(0.1)
    assert near and far and far[0].intensity > near[0].intensity  # same motion is a bigger swing for a small (far) hand


# ---- switch detector --------------------------------------------------------------
def test_switch_requires_both_fists_held_then_rearms():
    det = SwitchDetector(hold_s=1.0)
    fist_l, fist_r = make_hand(Side.LEFT, fingers(0)), make_hand(Side.RIGHT, fingers(0))
    open_r = make_hand(Side.RIGHT, fingers(5))
    fired = [det.update(fist_l, fist_r, i / 30) for i in range(45)]
    assert fired.count(True) == 1 and fired.index(True) >= 30
    assert det.progress == 1.0
    assert not det.update(fist_l, open_r, 2.0) and det.progress == 0.0
    assert not det.update(fist_l, None, 2.1)
    fired = [det.update(fist_l, fist_r, 3 + i / 30) for i in range(45)]
    assert fired.count(True) == 1


