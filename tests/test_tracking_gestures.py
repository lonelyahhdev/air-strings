
import pytest

from airstrings.config import GestureConfig
from airstrings.gestures.chord_selector import ChordSelector
from airstrings.gestures.pinch_detector import PinchDetector
from airstrings.gestures.switch_detector import SwitchDetector
from airstrings.gestures.types import Hit, Release, Trigger
from airstrings.tracking.features import count_fingers, finger_reach, finger_states, hand_scale, palm_center, pinch_ratios
from airstrings.tracking.synthetic import fingers, make_hand
from airstrings.tracking.types import LM, Side


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("n", range(6))
def test_finger_counting(side, n):
    assert count_fingers(make_hand(side, fingers(n))) == n


def test_finger_state_identity():
    hand = make_hand(Side.LEFT, (False, True, False, False, True))
    assert finger_states(hand) == (False, True, False, False, True)


def test_scale_and_palm():
    assert hand_scale(make_hand(Side.RIGHT, length=0.1)) == pytest.approx(0.1, rel=0.05)
    x, y = palm_center(make_hand(Side.RIGHT, center=(0.3, 0.6)))
    assert (x, y) == pytest.approx((0.3, 0.6), abs=0.005)


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


# ---- pinch detector ---------------------------------------------------------------
TAP = [0.0, 0.55, 1.0, 1.0, 0.55, 0.0]  # fingers close over two frames, then open
FINGER_KIND = {"index": Hit.CHORD, "middle": Hit.BASS, "ring": Hit.ARP}


def pinch_run(det, amounts, finger="index", side=Side.RIGHT, fps=30, t0=0.0, length=0.15):
    events = []
    for i, a in enumerate(amounts):
        hand = make_hand(side, fingers(5), (0.7, 0.6), length=length, pinch=finger, pinch_amount=a)
        events += det.update(hand, t0 + i / fps)
    return events


def triggers(events):
    return [e for e in events if isinstance(e, Trigger)]


def releases(events):
    return [e for e in events if isinstance(e, Release)]


def test_pinch_ratios_geometry():
    open_hand = make_hand(Side.RIGHT, fingers(5))
    assert min(pinch_ratios(open_hand)) > 1.0
    closed = make_hand(Side.RIGHT, fingers(5), pinch="middle")
    r = pinch_ratios(closed)
    assert r[1] == pytest.approx(0.0, abs=0.02) and r[0] > 0.2 and r[2] > 0.2


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("finger", ["index", "middle", "ring"])
def test_each_fingertip_plays_its_own_hit(side, finger):
    ev = pinch_run(PinchDetector(GestureConfig()), TAP, finger, side)
    assert [t.kind for t in triggers(ev)] == [FINGER_KIND[finger]]
    assert len(releases(ev)) == 1 and releases(ev)[0].kind is FINGER_KIND[finger]
    assert 0 < triggers(ev)[0].intensity <= 1


@pytest.mark.parametrize("finger", ["index", "middle", "ring"])
def test_slow_approach_is_not_misread_as_the_neighbouring_finger(finger):
    # the thumb passes the index finger on its way to the middle/ring fingertip
    ramp = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0]
    ev = pinch_run(PinchDetector(GestureConfig()), ramp, finger)
    assert [t.kind for t in triggers(ev)] == [FINGER_KIND[finger]]


def test_hold_reports_how_long_and_taps_report_short():
    det = PinchDetector(GestureConfig())
    held = releases(pinch_run(det, [0.0, 0.55] + [1.0] * 15 + [0.55, 0.0]))
    assert held[0].held_s == pytest.approx(0.5, abs=0.1)
    tap = releases(pinch_run(PinchDetector(GestureConfig()), TAP))
    assert tap[0].held_s < 0.15


def test_one_pinch_at_a_time_and_retrigger_after_release():
    det = PinchDetector(GestureConfig())
    ev = pinch_run(det, TAP * 3)
    assert len(triggers(ev)) == 3 and len(releases(ev)) == 3
    assert [type(e) for e in ev] == [Trigger, Release] * 3


def test_open_hand_and_relaxed_wobble_do_nothing():
    det = PinchDetector(GestureConfig())
    assert pinch_run(det, [0.0, 0.1, 0.3, 0.2, 0.4, 0.3, 0.1, 0.0] * 3) == []


def test_hysteresis_ignores_jitter_while_pinched():
    det = PinchDetector(GestureConfig())
    wobble = [0.0, 0.55] + [1.0, 0.97, 1.0, 0.94, 0.99, 0.96, 1.0] + [0.55, 0.0]
    ev = pinch_run(det, wobble)
    assert len(triggers(ev)) == 1 and len(releases(ev)) == 1


def test_closed_fist_is_not_a_pinch():
    det = PinchDetector(GestureConfig())
    fist = make_hand(Side.RIGHT, fingers(0), (0.7, 0.6), pinch="index", pinch_amount=1.0)
    assert finger_reach(fist, LM.INDEX_TIP) < GestureConfig().pinch_min_reach
    assert [e for i in range(10) for e in det.update(fist, i / 30)] == []


def test_pinky_pinch_is_reserved_and_not_misread_as_ring():
    det = PinchDetector(GestureConfig())
    assert pinch_run(det, TAP * 2, "pinky") == []


def test_thumb_between_two_fingertips_is_ambiguous_so_nothing_fires():
    hand = make_hand(Side.RIGHT, fingers(5), (0.7, 0.6))
    lm = hand.landmarks.copy()
    lm[LM.THUMB_TIP] = (lm[LM.INDEX_TIP] + lm[LM.MIDDLE_TIP]) / 2
    det = PinchDetector(GestureConfig())
    ev = [e for i in range(6) for e in det.update(type(hand)(hand.side, lm, 1.0, hand.aspect), i / 30)]
    assert triggers(ev) == []


def test_intensity_follows_closing_speed_with_a_floor():
    cfg = GestureConfig()
    fast = triggers(pinch_run(PinchDetector(cfg), TAP))[0].intensity
    slow = triggers(pinch_run(PinchDetector(cfg), [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.95, 1.0, 1.0]))[0].intensity
    assert fast > slow >= cfg.min_intensity
    assert fast == pytest.approx(1.0, abs=0.01)
    # a hand that first appears already pinched has no closing speed: softest hit
    det = PinchDetector(cfg)
    ev = pinch_run(det, [1.0, 1.0, 1.0])
    assert triggers(ev)[0].intensity == pytest.approx(cfg.min_intensity)


def test_works_for_small_distant_hands():
    ev = pinch_run(PinchDetector(GestureConfig()), TAP, "ring", length=0.06)
    assert [t.kind for t in triggers(ev)] == [Hit.ARP]


def test_hand_leaving_mid_pinch_releases():
    det = PinchDetector(GestureConfig())
    ev = pinch_run(det, [0.0, 0.55, 1.0, 1.0])
    assert len(triggers(ev)) == 1 and not releases(ev)
    gone = det.update(None, 0.2)
    assert [type(e) for e in gone] == [Release] and det.ratios is None
    assert det.update(None, 0.3) == []


def test_trigger_position_is_the_thumb():
    ev = pinch_run(PinchDetector(GestureConfig()), TAP)
    x, y = triggers(ev)[0].pos
    assert 0.5 < x < 0.95 and 0.2 < y < 0.8


def test_nudge_changes_sensitivity_and_keeps_gap():
    det = PinchDetector(GestureConfig())
    gap = det.off - det.on
    det.nudge(0.05)
    assert det.on == pytest.approx(0.35) and det.off - det.on == pytest.approx(gap)
    for _ in range(100):
        det.nudge(0.05)
    assert det.on == pytest.approx(0.6)
    for _ in range(100):
        det.nudge(-0.05)
    assert det.on == pytest.approx(0.12)


def test_easier_threshold_lets_looser_pinches_through():
    loose = [0.0, 0.5, 0.7, 0.7, 0.5, 0.0]  # fingertips come close (~0.4 hand-lengths), but don't touch
    assert triggers(pinch_run(PinchDetector(GestureConfig()), loose)) == []
    easy = PinchDetector(GestureConfig())
    easy.nudge(0.15)
    assert len(triggers(pinch_run(easy, loose))) == 1


def test_velocity_property_range():
    assert Trigger(0, Hit.CHORD, 0.0, (0, 0)).velocity == 24
    assert Trigger(0, Hit.CHORD, 1.0, (0, 0)).velocity == 127


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


