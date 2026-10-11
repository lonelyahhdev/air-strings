"""On-screen guidance drawn with OpenCV onto the (mirrored) camera frame.

Layout, designed so the player only has to look at the picture, not read docs:

* chord hand side: a ladder "1 finger = C, 2 = F ..." with the active row lit,
  plus a dashed line above which the hand adds a 7th;
* playing hand side: the three fingertips are labelled CHORD / BASS / ARP and a
  line from the thumb to each brightens as they get close; a ring flashes where a
  pinch registers;
* top bar: instrument tabs, key, mode; bottom: the upcoming sequence ribbon.
"""

from __future__ import annotations

import cv2
import numpy as np

from airstrings.engine.session import Snapshot
from airstrings.gestures.types import Hit
from airstrings.tracking.types import LM, Side

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]  # fmt: skip

# BGR
CHORD_COLOR = (255, 170, 60)
PLAY_COLOR = (60, 150, 255)
ACCENT = (80, 220, 140)
TEXT = (245, 245, 245)
DIM = (150, 150, 150)
PANEL = (28, 28, 28)
FONT = cv2.FONT_HERSHEY_DUPLEX
FLASH_S = 0.3
# (fingertip landmark, what pinching it plays, label)
PINCH_FINGERS = (
    (LM.INDEX_TIP, Hit.CHORD, "CHORD"),
    (LM.MIDDLE_TIP, Hit.BASS, "BASS"),
    (LM.RING_TIP, Hit.ARP, "ARP"),
)
PINCH_ROWS = ("thumb + index", "thumb + middle", "thumb + ring")


def _panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.55) -> None:
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(roi, PANEL), alpha, roi, 1 - alpha, 0, dst=roi)


def _text(img, s, org, scale=0.6, color=TEXT, thick=1) -> None:
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _dashed_hline(img, y: int, x0: int, x1: int, color, thick=1, dash=14) -> None:
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y), (min(x + dash, x1), y), color, thick, cv2.LINE_AA)


class Overlay:
    def draw(self, img: np.ndarray, snap: Snapshot) -> np.ndarray:
        h, w = img.shape[:2]
        self._skeletons(img, snap)
        self._pinch_marks(img, snap, w, h)
        self._play_panel(img, snap, w)
        self._chord_ladder(img, snap, w, h)
        self._header(img, snap, w)
        self._ribbon(img, snap, w, h)
        self._switch_ring(img, snap, w, h)
        return img

    # -- pieces ----------------------------------------------------------------------
    def _skeletons(self, img, snap: Snapshot) -> None:
        h, w = img.shape[:2]
        for hand in snap.hands:
            color = CHORD_COLOR if hand.side is snap.harmony_side else PLAY_COLOR
            pts = (hand.landmarks[:, :2] * (w, h)).astype(int)
            for a, b in BONES:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(p), 3, TEXT, -1, cv2.LINE_AA)

    def _pinch_marks(self, img, snap: Snapshot, w: int, h: int) -> None:
        hand = next((hd for hd in snap.hands if hd.side is not snap.harmony_side), None)
        if hand is not None:
            pts = (hand.landmarks[:, :2] * (w, h)).astype(int)
            thumb = (int(pts[LM.THUMB_TIP][0]), int(pts[LM.THUMB_TIP][1]))
            ratios = snap.pinch_ratios or (9.0,) * 4
            for i, (tip_idx, kind, label) in enumerate(PINCH_FINGERS):
                tip = (int(pts[tip_idx][0]), int(pts[tip_idx][1]))
                near = min(max(1.0 - (ratios[i] - snap.pinch_on) / 0.8, 0.0), 1.0)  # 0 far .. 1 at the trigger point
                active = snap.pinch_active is kind
                color = ACCENT if active else tuple(int(d + (p - d) * near) for d, p in zip(DIM, PLAY_COLOR))
                cv2.line(img, thumb, tip, color, 1 + int(3 * near), cv2.LINE_AA)
                cv2.circle(img, tip, 16 if active else 11, color, -1 if active else 2, cv2.LINE_AA)
                (tw, _), _ = cv2.getTextSize(label, FONT, 0.55, 1)
                lift = 44 if i == 1 else 22  # stagger the middle label so neighbours don't collide
                _text(img, label, (tip[0] - tw // 2, tip[1] - lift), 0.55, ACCENT if active else TEXT, 2 if active else 1)
        trig = snap.last_trigger
        age = snap.t - trig.t if trig else 1e9
        if trig and age < FLASH_S:
            frac = age / FLASH_S
            centre = (int(trig.pos[0] * w), int(trig.pos[1] * h))
            cv2.circle(img, centre, int(20 + 60 * frac), ACCENT, max(1, int(7 * (1 - frac))), cv2.LINE_AA)

    def _play_panel(self, img, snap: Snapshot, w: int) -> None:
        x0 = w - 236 if snap.harmony_side is Side.LEFT else 16
        _panel(img, x0 - 8, 90, x0 + 220, 90 + 38 * 3 + 112)
        _text(img, "PLAYING HAND", (x0, 118), 0.55, PLAY_COLOR)
        for i, ((_tip, kind, label), what) in enumerate(zip(PINCH_FINGERS, PINCH_ROWS)):
            y = 150 + i * 38
            on = snap.pinch_active is kind
            if on:
                cv2.rectangle(img, (x0 - 4, y - 24), (x0 + 210, y + 8), PLAY_COLOR, -1)
            col = (20, 20, 20) if on else TEXT
            _text(img, what.replace("thumb + ", "+ "), (x0, y), 0.55, col)
            _text(img, label, (x0 + 120, y), 0.7, col, 2)
        _text(img, "hold = sustain", (x0, 150 + 3 * 38 + 4), 0.5, DIM)
        _text(img, "let go = stop", (x0, 150 + 3 * 38 + 26), 0.5, DIM)
        trig = snap.last_trigger
        if trig and snap.t - trig.t < 0.8:  # keep the strength readout visible long enough to read
            _text(img, f"{trig.kind.value.upper()}  strength {trig.velocity}", (x0, 90 + 38 * 3 + 112 + 30), 0.6, ACCENT, 2)

    def _chord_ladder(self, img, snap: Snapshot, w: int, h: int) -> None:
        left = snap.harmony_side is Side.LEFT
        x0 = 16 if left else w - 236
        if snap.mode != "free":  # the chord hand does nothing here: say so instead of showing a dead ladder
            _panel(img, x0 - 8, 90, x0 + 220, 168)
            _text(img, "CHORDS ARE AUTOMATIC", (x0, 118), 0.55, CHORD_COLOR)
            _text(img, "just swing to play the", (x0, 144), 0.5, DIM)
            _text(img, "next note in the ribbon", (x0, 162), 0.5, DIM)
            return
        rows = len(snap.chord_ladder) + 1
        _panel(img, x0 - 8, 90, x0 + 220, 90 + 38 * rows + 52)
        _text(img, "CHORD HAND", (x0, 118), 0.55, CHORD_COLOR)
        active = snap.intent.fingers if snap.intent.degree is not None else 0
        for i, (fingers, name) in enumerate(snap.chord_ladder):
            y = 150 + i * 38
            on = fingers == active
            if on:
                cv2.rectangle(img, (x0 - 4, y - 24), (x0 + 210, y + 8), CHORD_COLOR, -1)
            col = (20, 20, 20) if on else TEXT
            _text(img, f"{fingers} finger{'s' if fingers > 1 else ' '}", (x0, y), 0.6, col)
            _text(img, name, (x0 + 130, y), 0.75, col, 2)
        fy = 150 + len(snap.chord_ladder) * 38
        _text(img, "fist = silence", (x0, fy), 0.55, DIM)
        zone_y = int(snap.seventh_zone_y * h)
        xa, xb = (0, w // 2) if left else (w // 2, w)
        _dashed_hline(img, zone_y, xa, xb, CHORD_COLOR)
        label_x = 250 if left else xa + 16  # keep clear of the ladder panel
        _text(img, "raise hand above this line for 7ths", (label_x, zone_y - 8), 0.5, CHORD_COLOR)
        if snap.intent.extended:
            _text(img, "7th ON", (label_x, zone_y + 26), 0.7, ACCENT, 2)

    def _header(self, img, snap: Snapshot, w: int) -> None:
        _panel(img, 0, 0, w, 64, 0.65)
        x = 16
        for name in snap.instruments:
            on = name == snap.instrument
            label = name.upper()
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.7, 2 if on else 1)
            if on:
                cv2.rectangle(img, (x - 8, 14), (x + tw + 8, 48), ACCENT, -1)
            _text(img, label, (x, 40), 0.7, (20, 20, 20) if on else DIM, 2 if on else 1)
            x += tw + 28
        info = f"Key: {snap.key_name}    Mode: {snap.mode.upper()}"
        if snap.chord_name:
            info += f"    Chord: {snap.chord_name}"
        _text(img, info, (x + 20, 40), 0.7)
        if snap.status:
            (tw, _), _ = cv2.getTextSize(snap.status, FONT, 0.55, 1)
            _text(img, snap.status, (w - tw - 16, 40), 0.55, ACCENT)

    def _ribbon(self, img, snap: Snapshot, w: int, h: int) -> None:
        _panel(img, 0, h - 84, w, h, 0.65)
        if snap.mode != "sequence":
            hint = "1/2 instrument   m sequence mode   ,/. transpose   n major/minor   [ ] pinch sensitivity   q quit"
            _text(img, hint, (16, h - 32), 0.55, DIM)
            return
        _text(img, f"SEQUENCE {snap.sequence_name} - highlighted note plays next".replace("  ", " "), (16, h - 54), 0.5, DIM)
        x = 16
        for i, label in enumerate(snap.sequence_upcoming):
            (tw, _), _ = cv2.getTextSize(label, FONT, 1.0 if i == 0 else 0.7, 2)
            if i == 0:
                cv2.rectangle(img, (x - 8, h - 50), (x + tw + 8, h - 10), ACCENT, -1)
            _text(img, label, (x, h - 20), 1.0 if i == 0 else 0.7, (20, 20, 20) if i == 0 else TEXT, 2)
            x += tw + 30
        if not snap.sequence_upcoming:
            _text(img, "sequence finished - press r to restart", (16, h - 20), 0.7, ACCENT)

    def _switch_ring(self, img, snap: Snapshot, w: int, h: int) -> None:
        if snap.switch_progress <= 0:
            return
        c = (w // 2, h // 2)
        cv2.ellipse(img, c, (60, 60), -90, 0, int(360 * snap.switch_progress), ACCENT, 8, cv2.LINE_AA)
        _text(img, "switching instrument", (c[0] - 105, c[1] + 95), 0.6, ACCENT)
