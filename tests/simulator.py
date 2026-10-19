"""A synthetic Minesweeper game + renderer used for vision/agent tests (no real screen needed)."""
from __future__ import annotations

import random

import cv2
import numpy as np

COLORS = {1: (255, 0, 0), 2: (0, 128, 0), 3: (0, 0, 255), 4: (128, 0, 0),   # BGR
          5: (0, 0, 128), 6: (128, 128, 0), 7: (0, 0, 0), 8: (128, 128, 128)}


class Game:
    def __init__(self, rows=9, cols=9, mines=10, seed=1):
        self.rows, self.cols, self.n_mines = rows, cols, mines
        self.rng = random.Random(seed)
        self.mines: set = set()
        self.revealed = np.zeros((rows, cols), bool)
        self.flags = np.zeros((rows, cols), bool)
        self.started = False
        self.lost = False
        self.won = False

    def neighbors(self, r, c):
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]

    def number(self, r, c):
        return sum(q in self.mines for q in self.neighbors(r, c))

    def reveal(self, r, c):
        if self.lost or self.won or self.flags[r, c] or self.revealed[r, c]:
            return
        if not self.started:  # first click is always safe
            self.started = True
            cells = [(i, j) for i in range(self.rows) for j in range(self.cols)
                     if (i, j) != (r, c) and (i, j) not in self.neighbors(r, c)]
            self.mines = set(self.rng.sample(cells, self.n_mines))
        if (r, c) in self.mines:
            self.lost = True
            self.revealed[r, c] = True
            return
        stack = [(r, c)]
        while stack:
            i, j = stack.pop()
            if self.revealed[i, j]:
                continue
            self.revealed[i, j] = True
            if self.number(i, j) == 0:
                stack.extend(q for q in self.neighbors(i, j) if not self.revealed[q] and not self.flags[q])
        safe_total = self.rows * self.cols - self.n_mines
        if int(self.revealed.sum()) == safe_total:
            self.won = True

    def flag(self, r, c):
        if not self.revealed[r, c] and not self.lost:
            self.flags[r, c] = not self.flags[r, c]

    def render(self, cell=32, margin=(60, 90), canvas=(900, 700), style="classic") -> np.ndarray:
        h, w = canvas[1], canvas[0]
        img = np.full((h, w, 3), 245, np.uint8)
        cv2.putText(img, "Minesweeper - friend's site", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
        cv2.rectangle(img, (20, 45), (160, 75), (200, 180, 120), -1)
        x0, y0 = margin
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = x0 + c * cell, y0 + r * cell
                p1, p2 = (x, y), (x + cell - 1, y + cell - 1)
                shown = self.revealed[r, c] or (self.lost and (r, c) in self.mines)
                if not shown:
                    cv2.rectangle(img, p1, p2, (190, 190, 190), -1)
                    cv2.line(img, p1, (x + cell - 2, y), (255, 255, 255), 3)
                    cv2.line(img, p1, (x, y + cell - 2), (255, 255, 255), 3)
                    cv2.line(img, (x + cell - 1, y + 1), p2, (110, 110, 110), 3)
                    cv2.line(img, (x + 1, y + cell - 1), p2, (110, 110, 110), 3)
                    if self.flags[r, c]:
                        pts = np.array([[x + cell * .3, y + cell * .25], [x + cell * .75, y + cell * .4],
                                        [x + cell * .3, y + cell * .55]], np.int32)
                        cv2.fillPoly(img, [pts], (0, 0, 220))
                        cv2.line(img, (int(x + cell * .3), int(y + cell * .25)),
                                 (int(x + cell * .3), int(y + cell * .8)), (0, 0, 0), 2)
                else:
                    cv2.rectangle(img, p1, p2, (165, 165, 165), -1)
                    cv2.rectangle(img, p1, p2, (120, 120, 120), 1)
                    if (r, c) in self.mines:
                        cv2.circle(img, (x + cell // 2, y + cell // 2), int(cell * .3), (0, 0, 0), -1)
                    else:
                        n = self.number(r, c)
                        if n:
                            sc = cell / 40
                            (tw, th), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, sc * 1.0, 2)
                            cv2.putText(img, str(n), (x + (cell - tw) // 2, y + (cell + th) // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, sc * 1.0, COLORS[n], 2, cv2.LINE_AA)
        self.origin = (x0, y0, cell)
        return img

    def cell_at(self, px, py, cell=32, margin=(60, 90)):
        c, r = (px - margin[0]) // cell, (py - margin[1]) // cell
        return (int(r), int(c)) if 0 <= r < self.rows and 0 <= c < self.cols else None
