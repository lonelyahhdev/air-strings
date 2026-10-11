"""Procedural hands for tests, demos and headless runs (no camera needed)."""

from __future__ import annotations

import numpy as np

from airstrings.tracking.types import LM, Hand, Side


def make_hand(
    side: Side,
    extended: tuple[bool, bool, bool, bool, bool] = (True,) * 5,
    center: tuple[float, float] = (0.5, 0.5),
    length: float = 0.15,
    aspect: float = 16 / 9,
    pinch: str | None = None,
    pinch_amount: float = 1.0,
) -> Hand:
    """A palm-forward, fingers-up hand whose palm centre is near ``center``.

    ``length`` is wrist -> middle knuckle in image-height units.  ``pinch`` is
    ``"index"``, ``"middle"``, ``"ring"`` or ``"pinky"``: the thumb tip moves toward
    that fingertip (``pinch_amount`` 0 = where it was, 1 = touching).
    """
    cx, cy = center
    L = length
    s = 1.0 if side is Side.RIGHT else -1.0  # thumb side in mirrored view
    pts = np.zeros((21, 2), dtype=np.float32)  # in image-height units, x not yet / aspect

    # wrist -> middle knuckle is L, and the mean of wrist + 4 knuckles (palm_center) is cy
    wrist = np.array([cx * aspect, cy + 0.8 * L])
    mcp_y = cy - 0.2 * L
    pts[LM.WRIST] = wrist
    knuckle_x = {LM.INDEX_MCP: 0.42, LM.MIDDLE_MCP: 0.14, LM.RING_MCP: -0.14, LM.PINKY_MCP: -0.42}
    finger_len = {LM.INDEX_MCP: 0.95, LM.MIDDLE_MCP: 1.0, LM.RING_MCP: 0.92, LM.PINKY_MCP: 0.75}
    for mcp, flag in zip(knuckle_x, extended[1:]):
        x = cx * aspect + s * knuckle_x[mcp] * L
        f = finger_len[mcp]
        pts[mcp] = (x, mcp_y)
        if flag:
            pts[mcp + 1] = (x, mcp_y - 0.45 * L * f)
            pts[mcp + 2] = (x, mcp_y - 0.70 * L * f)
            pts[mcp + 3] = (x, mcp_y - 0.90 * L * f)
        else:  # curled toward the palm: tip ends up below the knuckle
            pts[mcp + 1] = (x, mcp_y - 0.25 * L)
            pts[mcp + 2] = (x, mcp_y + 0.02 * L)
            pts[mcp + 3] = (x, mcp_y + 0.22 * L)

    t_base = wrist + np.array([s * 0.25 * L, -0.15 * L])
    pts[LM.THUMB_CMC] = t_base
    pts[LM.THUMB_MCP] = t_base + np.array([s * 0.25 * L, -0.25 * L])
    if extended[0]:
        pts[LM.THUMB_IP] = pts[LM.THUMB_MCP] + np.array([s * 0.30 * L, -0.15 * L])
        pts[LM.THUMB_TIP] = pts[LM.THUMB_MCP] + np.array([s * 0.55 * L, -0.25 * L])
    else:  # folded across the palm toward the pinky
        pts[LM.THUMB_IP] = pts[LM.THUMB_MCP] + np.array([-s * 0.10 * L, -0.15 * L])
        pts[LM.THUMB_TIP] = pts[LM.THUMB_MCP] + np.array([-s * 0.35 * L, -0.10 * L])

    if pinch is not None:
        tip = {"index": LM.INDEX_TIP, "middle": LM.MIDDLE_TIP, "ring": LM.RING_TIP, "pinky": LM.PINKY_TIP}[pinch]
        amount = min(max(pinch_amount, 0.0), 1.0)
        start = pts[LM.THUMB_TIP].copy()
        pts[LM.THUMB_TIP] = start + (pts[tip] - start) * amount
        pts[LM.THUMB_IP] = pts[LM.THUMB_IP] + (pts[tip] - pts[LM.THUMB_IP]) * amount * 0.6

    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 0] = pts[:, 0] / aspect
    lm[:, 1] = pts[:, 1]
    return Hand(side, lm, 1.0, aspect)


def fingers(n: int) -> tuple[bool, bool, bool, bool, bool]:
    """Extended-flags for a natural count: 1 = index, 2 = +middle ... 5 = all + thumb."""
    if not 0 <= n <= 5:
        raise ValueError("n must be 0..5")
    if n == 5:
        return (True,) * 5
    return (False,) + tuple(i < n for i in range(4))  # type: ignore[return-value]
