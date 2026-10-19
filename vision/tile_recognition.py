"""Classify each grid cell from pixels. Style-agnostic: it *learns* what a covered tile looks like.

Per cell we compute a 15-value signature (mean colour of the 4 edge strips + median background
colour) and compare it with learned 'covered' and 'revealed' signatures. Revealed cells are then
read by finding 'ink' pixels (differing from the cell's own background) and matching their colour
(and, if taught, shape) against the digit samples in the profile.

Limits: digits are told apart by colour/shape samples, so unusual themes need `teach_digit`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from solver.board_state import COVERED, FLAG, MINE
from .board_detection import GridGeometry

S = 24  # normalised patch size
COVERED_REV_THRESHOLD = 12.0

# Common Minesweeper palette (BGR). Only a default starting point; override by teaching digits.
DEFAULT_DIGITS = {"1": (255, 0, 0), "2": (0, 128, 0), "3": (0, 0, 255), "4": (128, 0, 0),
                  "5": (0, 0, 128), "6": (128, 128, 0), "7": (0, 0, 0), "8": (110, 110, 110)}


@dataclass
class TileProfile:
    name: str = "default"
    covered_sig: list | None = None
    revealed_sig: list | None = None
    ink_threshold: float = 30.0
    digits: dict = field(default_factory=lambda: {k: [{"color": list(v), "mask": None}]
                                                  for k, v in DEFAULT_DIGITS.items()})

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> "TileProfile":
        p = cls()
        for k, v in json.loads(path.read_text()).items():
            setattr(p, k, v)
        return p


@dataclass
class RecognitionResult:
    grid: np.ndarray          # int grid using solver.board_state codes
    confidence: np.ndarray    # float [0,1] per cell
    notes: list = field(default_factory=list)

    @property
    def min_confidence(self) -> float:
        return float(self.confidence.min()) if self.confidence.size else 0.0

    @property
    def mean_confidence(self) -> float:
        return float(self.confidence.mean()) if self.confidence.size else 0.0

    def low_cells(self, threshold: float) -> list[tuple[int, int]]:
        return [(int(r), int(c)) for r, c in np.argwhere(self.confidence < threshold)]

    def is_reliable(self, threshold: float) -> bool:
        return self.confidence.size > 0 and self.min_confidence >= threshold


class ProfileMissingError(RuntimeError):
    pass


INSET = 0.10  # sample slightly inside each cell so small grid-alignment errors never leak neighbours in


def _patch(frame: np.ndarray, rect) -> np.ndarray | None:
    x0, y0, x1, y1 = rect
    dx, dy = int(round((x1 - x0) * INSET)), int(round((y1 - y0) * INSET))
    x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy
    h, w = frame.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return cv2.resize(frame[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_AREA)


def _band(f: np.ndarray) -> np.ndarray:
    """Pixels from the outermost 2px of the (inset) patch: background only, never digit ink."""
    return np.concatenate([f[1:3].reshape(-1, 3), f[-3:-1].reshape(-1, 3),
                           f[:, 1:3].reshape(-1, 3), f[:, -3:-1].reshape(-1, 3)])


def _dominant_color(f: np.ndarray) -> np.ndarray:
    """Most common colour in the patch = the tile body (digits/flags cover less than half of it)."""
    q = (f.reshape(-1, 3) // 8).astype(np.int32)
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    vals, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return f.reshape(-1, 3)[inv == int(np.argmax(counts))].mean(axis=0)


def signature(p: np.ndarray) -> np.ndarray:
    """15 numbers describing the tile *style*, deliberately blind to digits/flags in the middle:
    mean colour of the four 3x3 corner blocks plus the dominant (body) colour."""
    f = p.astype(np.float32)
    corners = [f[1:4, 1:4], f[1:4, -4:-1], f[-4:-1, 1:4], f[-4:-1, -4:-1]]
    return np.concatenate([c.reshape(-1, 3).mean(axis=0) for c in corners] + [_dominant_color(f)])


def _ink(p: np.ndarray, thr: float):
    """Return (mask over the centre 12x12, bg colour, centre patch)."""
    f = p.astype(np.float32)
    bg = _dominant_color(f)
    centre = f[6:18, 6:18]
    diff = np.abs(centre - bg).max(axis=2)
    return diff > thr, diff, bg, centre


def is_mine_blob(mask: np.ndarray, dark: bool) -> bool:
    """A mine is a big, dark, compact blob; digits and flags are thinner (lower bbox fill)."""
    n = int(mask.sum())
    if n < 80 or not dark:
        return False
    ys, xs = np.nonzero(mask)
    fill = n / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1))
    return fill > 0.8


class TileRecognizer:
    def __init__(self, profile: TileProfile | None = None):
        self.profile = profile or TileProfile()

    # ------------------------------------------------------------------ learning
    def learn_covered(self, frame: np.ndarray, geo: GridGeometry) -> int:
        """Call on a *fresh* board (every cell covered). Returns the number of cells sampled."""
        sigs = [signature(p) for r in range(geo.rows) for c in range(geo.cols)
                if (p := _patch(frame, geo.cell_rect(r, c))) is not None]
        if not sigs:
            raise ProfileMissingError("No readable cells inside the frame.")
        arr = np.array(sigs)
        spread = float(np.percentile(np.abs(arr - np.median(arr, axis=0)).mean(axis=1), 85))
        if spread > 12:
            raise ProfileMissingError(
                "The cells do not look identical, so this does not seem to be a fresh board. "
                "Start a new game and learn again.")
        self.profile.covered_sig = np.median(arr, axis=0).tolist()
        self.profile.revealed_sig = None
        return len(sigs)

    def teach_digit(self, frame: np.ndarray, geo: GridGeometry, cell: tuple[int, int], digit: int) -> None:
        p = _patch(frame, geo.cell_rect(*cell))
        if p is None:
            raise ValueError("Cell is outside the frame")
        mask, diff, _, centre = _ink(p, self.profile.ink_threshold)
        if mask.sum() < 3:
            raise ValueError("No digit ink found in that cell")
        color, shape = self._ink_color(mask, diff, centre), mask.astype(float)
        self.profile.digits.setdefault(str(digit), [])
        # a taught sample replaces untaught defaults for this digit
        self.profile.digits[str(digit)] = [s for s in self.profile.digits[str(digit)] if s["mask"] is not None]
        self.profile.digits[str(digit)].append({"color": color.tolist(), "mask": shape.tolist()})

    # ------------------------------------------------------------------ recognition
    @staticmethod
    def _ink_color(mask, diff, centre) -> np.ndarray:
        d = diff[mask]
        sel = d >= np.percentile(d, 60)
        return centre[mask][sel].mean(axis=0)

    def _digit(self, mask, diff, centre) -> tuple[int, float]:
        color = self._ink_color(mask, diff, centre)
        scores = []
        for dg, samples in self.profile.digits.items():
            best = 9.0
            for s in samples:
                dc = float(np.linalg.norm(color - np.array(s["color"]))) / 441.0
                if s["mask"] is not None:
                    ds = float(np.abs(mask.astype(float) - np.array(s["mask"])).mean())
                else:
                    ds = 0.15
                best = min(best, dc * 3 + ds)
            scores.append((best, int(dg)))
        scores.sort()
        d1, d2 = scores[0][0], scores[1][0]
        conf = max(0.0, min(1.0, (d2 - d1) / 0.4))  # absolute margin between best and runner-up
        return scores[0][1], conf

    def recognize(self, frame: np.ndarray, geo: GridGeometry) -> RecognitionResult:
        if self.profile.covered_sig is None:
            raise ProfileMissingError("No tile profile yet: calibrate on a fresh board first.")
        rows, cols = geo.rows, geo.cols
        grid = np.full((rows, cols), COVERED, dtype=int)
        conf = np.zeros((rows, cols), dtype=float)
        patches: dict = {}
        sigs: dict = {}
        for r in range(rows):
            for c in range(cols):
                p = _patch(frame, geo.cell_rect(r, c))
                if p is not None:
                    patches[(r, c)], sigs[(r, c)] = p, signature(p)
        cov = np.array(self.profile.covered_sig)
        dc = {k: float(np.abs(s - cov).mean()) for k, s in sigs.items()}
        if self.profile.revealed_sig is None:
            far = [sigs[k] for k, d in dc.items() if d > 2 * COVERED_REV_THRESHOLD]
            if far:
                self.profile.revealed_sig = np.median(np.array(far), axis=0).tolist()
        rev = None if self.profile.revealed_sig is None else np.array(self.profile.revealed_sig)
        notes = []
        for (r, c), p in patches.items():
            d_cov = dc[(r, c)]
            if rev is not None:
                d_rev = float(np.abs(sigs[(r, c)] - rev).mean())
                is_cov = d_cov < d_rev
                cc = abs(d_cov - d_rev) / (d_cov + d_rev + 1e-6)
                cc = min(1.0, cc * 1.5)
            else:
                is_cov = d_cov < COVERED_REV_THRESHOLD
                cc = (max(0.0, min(1.0, (COVERED_REV_THRESHOLD - d_cov) / 6.0)) if is_cov
                      else max(0.0, min(1.0, (d_cov - COVERED_REV_THRESHOLD) / 6.0 + 0.5)))
            mask, diff, bg, centre = _ink(p, self.profile.ink_threshold)
            n_ink = int(mask.sum())
            dark = n_ink > 0 and float(centre[mask].mean()) < 90
            if is_cov:
                if is_mine_blob(mask, dark):
                    grid[r, c], ic = MINE, 0.8
                elif n_ink >= 10:
                    grid[r, c], ic = FLAG, 1.0
                elif n_ink >= 4:
                    grid[r, c], ic = FLAG, 0.4  # too small to trust
                else:
                    grid[r, c], ic = COVERED, 1.0 if n_ink == 0 else 0.6
            else:
                if is_mine_blob(mask, dark):
                    grid[r, c], ic = MINE, 0.8
                elif n_ink >= 4:
                    grid[r, c], ic = self._digit(mask, diff, centre)
                else:
                    grid[r, c], ic = 0, 1.0 if n_ink == 0 else 0.6
            conf[r, c] = min(cc, ic)
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in patches:
                    notes.append(f"Cell ({r+1},{c+1}) lies outside the captured frame.")
        return RecognitionResult(grid, conf, notes)
