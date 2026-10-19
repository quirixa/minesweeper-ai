"""Find the Minesweeper grid in a screenshot using only pixel structure (no site-specific colours).

Idea: every grid has long, regularly spaced horizontal and vertical edges. We (1) isolate long
straight edge runs, (2) take the largest connected cluster of them as the board, and (3) measure the
cell pitch with autocorrelation of the edge profile. When that is unreliable the user calibrates by
hand (`geometry_from_manual`).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GridGeometry:
    x0: float
    y0: float
    cell_w: float
    cell_h: float
    rows: int
    cols: int
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return self.cell_w * self.cols

    @property
    def height(self) -> float:
        return self.cell_h * self.rows

    def cell_rect(self, r: int, c: int) -> tuple[int, int, int, int]:
        x1, y1 = self.x0 + c * self.cell_w, self.y0 + r * self.cell_h
        return int(round(x1)), int(round(y1)), int(round(x1 + self.cell_w)), int(round(y1 + self.cell_h))

    def cell_center(self, r: int, c: int) -> tuple[float, float]:
        return self.x0 + (c + 0.5) * self.cell_w, self.y0 + (r + 0.5) * self.cell_h

    def to_dict(self) -> dict:
        return dict(x0=self.x0, y0=self.y0, cell_w=self.cell_w, cell_h=self.cell_h,
                    rows=self.rows, cols=self.cols, confidence=self.confidence)

    @classmethod
    def from_dict(cls, d: dict) -> "GridGeometry":
        return cls(**d)


def geometry_from_manual(x0: float, y0: float, x1: float, y1: float, rows: int, cols: int) -> GridGeometry:
    if rows < 1 or cols < 1 or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid manual grid specification")
    return GridGeometry(x0, y0, (x1 - x0) / cols, (y1 - y0) / rows, rows, cols, 1.0)


def _clusters(profile: np.ndarray, frac: float = 0.5, gap: int = 4) -> list[float]:
    """Centres of runs of strong response in a 1-D line-count profile."""
    if profile.max() <= 0:
        return []
    idx = np.nonzero(profile > frac * profile.max())[0]
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def _regular_run(centers: list[float], lo: int, hi: int, tol: float = 0.25) -> list[float]:
    """Longest run of centres with near-constant spacing (the grid lines)."""
    if len(centers) < 3:
        return []
    d = np.diff(centers)
    ok = d[(d >= lo) & (d <= hi)]
    if len(ok) < 2:
        return []
    m = float(np.median(ok))
    best, cur = [], [centers[0]]
    for prev, c in zip(centers, centers[1:]):
        if abs((c - prev) - m) <= tol * m:
            cur.append(c)
        else:
            best = cur if len(cur) > len(best) else best
            cur = [c]
    best = cur if len(cur) > len(best) else best
    return best if len(best) >= 3 else []


def _fit(centers: list[float]) -> tuple[float, float, float]:
    """Fit centre_k = a + k*p. The outer lines are biased by the bevel/background step, so the
    pitch and phase come from the interior lines when there are enough of them.
    Returns (a, p, rms residual / p)."""
    c = np.array(centers)
    k = np.arange(len(c))
    inner = slice(1, -1) if len(c) >= 5 else slice(None)
    p, _ = np.polyfit(k[inner], c[inner], 1)
    a = float(np.median(c[inner] - k[inner] * p))
    res = float(np.sqrt(np.mean((a + p * k[inner] - c[inner]) ** 2)) / p)
    return a, float(p), res


def detect_grid(frame: np.ndarray, min_cell: int = 10, max_cell: int = 140) -> GridGeometry | None:
    """Return the detected grid geometry in `frame` pixel coordinates, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 8, 24)
    k = max(2 * min_cell, 20)
    hor = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    ver = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
    lines = cv2.bitwise_or(hor, ver)
    if not lines.any():
        return None
    # Candidate regions = connected clusters of long edge runs. Try the biggest few: the board is
    # the one whose horizontal AND vertical lines are both regularly spaced.
    blob = cv2.dilate(lines, np.ones((min_cell + 5, min_cell + 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:4]
    best: GridGeometry | None = None
    for lab in order:
        comp = labels == lab
        ys, xs = np.nonzero(comp & (lines > 0))
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if x1 - x0 < 3 * min_cell or y1 - y0 < 3 * min_cell:
            continue
        hsub = (hor[y0:y1, x0:x1] > 0) & comp[y0:y1, x0:x1]
        vsub = (ver[y0:y1, x0:x1] > 0) & comp[y0:y1, x0:x1]
        rows_c = _regular_run(_clusters(hsub.sum(axis=1).astype(float)), min_cell, max_cell)
        cols_c = _regular_run(_clusters(vsub.sum(axis=0).astype(float)), min_cell, max_cell)
        if not rows_c or not cols_c:
            continue
        ay, py, ry = _fit(rows_c)
        ax, px, rx = _fit(cols_c)
        conf = max(0.0, 1.0 - 4 * max(ry, rx))
        if abs(px - py) / max(px, py) > 0.15:
            conf *= 0.5  # cells are nearly always square
        geo = GridGeometry(float(x0 + ax), float(y0 + ay), px, py, len(rows_c) - 1, len(cols_c) - 1, conf)
        cells = geo.rows * geo.cols
        if best is None or cells > best.rows * best.cols:
            best = geo
    return best
