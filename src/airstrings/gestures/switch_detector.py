"""Hold both fists to swap instrument."""

from __future__ import annotations

from airstrings.tracking.features import is_fist
from airstrings.tracking.types import Hand


class SwitchDetector:
    def __init__(self, hold_s: float = 1.0):
        self.hold_s = hold_s
        self._since: float | None = None
        self._fired = False
        self.progress = 0.0  # 0..1 for the on-screen ring

    def update(self, a: Hand | None, b: Hand | None, t: float) -> bool:
        """True exactly once per hold; release (open a hand) to re-arm."""
        if a is not None and b is not None and is_fist(a) and is_fist(b):
            if self._since is None:
                self._since = t
            self.progress = 1.0 if self._fired else min((t - self._since) / self.hold_s, 1.0)
            if not self._fired and t - self._since >= self.hold_s:
                self._fired = True
                return True
            return False
        self._since, self._fired, self.progress = None, False, 0.0
        return False
