"""Temporal consistency (agreement across frames) and board-change / stability detection."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .tile_recognition import RecognitionResult


class TemporalFilter:
    """A cell is trusted only if its value was identical in the last `window` recognitions."""

    def __init__(self, window: int = 3):
        self.window = max(1, window)
        self._hist: deque = deque(maxlen=self.window)

    def reset(self) -> None:
        self._hist.clear()

    def add(self, res: RecognitionResult) -> RecognitionResult:
        if self._hist and self._hist[-1].shape != res.grid.shape:
            self._hist.clear()
        self._hist.append(res.grid.copy())
        stable = np.ones(res.grid.shape, bool)
        for g in self._hist:
            stable &= (g == res.grid)
        if len(self._hist) < self.window:
            stable[:] = False if self.window > 1 else stable
        conf = np.where(stable, res.confidence, 0.0)
        notes = list(res.notes)
        if not stable.all():
            notes.append(f"{int((~stable).sum())} cell(s) changed between recent frames.")
        return RecognitionResult(res.grid, conf, notes)


def frame_change(a: np.ndarray, b: np.ndarray, pixel_threshold: int = 24) -> float:
    """Fraction of pixels that differ noticeably between two same-size frames (0 if sizes differ -> 1)."""
    if a.shape != b.shape:
        return 1.0
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float((cv2.absdiff(ga, gb) > pixel_threshold).mean())


class BoardChangeDetector:
    """Reports whether the board changed relative to the previous frame it saw."""

    def __init__(self, threshold: float = 0.0005):
        self.threshold = threshold
        self._prev: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> bool:
        changed = self._prev is None or frame_change(self._prev, frame) > self.threshold
        self._prev = frame.copy()
        return changed


class StabilityTracker:
    """True once `required` consecutive frames show no change (e.g. animation finished)."""

    def __init__(self, required: int = 2, threshold: float = 0.0005):
        self.required = max(1, required)
        self._det = BoardChangeDetector(threshold)
        self._still = 0

    def update(self, frame: np.ndarray) -> bool:
        if self._det.update(frame):
            self._still = 0
        else:
            self._still += 1
        return self._still >= self.required - 1 and (self._still > 0 or self.required == 1)
