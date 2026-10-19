"""Board representation shared by the vision layer and the solver."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

COVERED = -1   # unrevealed, unflagged
FLAG = -2      # flagged by the player
MINE = -3      # a revealed mine (game lost)

Cell = tuple[int, int]


@dataclass
class BoardState:
    grid: np.ndarray  # int array [rows, cols]; 0..8 revealed numbers, or COVERED/FLAG/MINE

    @classmethod
    def from_rows(cls, rows: list[str]) -> "BoardState":
        """Build a fixture from text: '#' covered, 'F' flag, '.' zero, '1'-'8', '*' mine."""
        m = {"#": COVERED, "F": FLAG, ".": 0, "*": MINE}
        return cls(np.array([[m[ch] if ch in m else int(ch) for ch in r] for r in rows], dtype=int))

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    def neighbors(self, r: int, c: int) -> list[Cell]:
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]

    def covered_cells(self) -> list[Cell]:
        return [(int(r), int(c)) for r, c in np.argwhere(self.grid == COVERED)]

    def count(self, value: int) -> int:
        return int((self.grid == value).sum())

    @property
    def revealed_count(self) -> int:
        return int(((self.grid >= 0)).sum())

    @property
    def flag_count(self) -> int:
        return self.count(FLAG)

    @property
    def is_lost(self) -> bool:
        return self.count(MINE) > 0

    def is_won(self, total_mines: int | None = None) -> bool:
        if self.is_lost:
            return False
        covered, flags = self.count(COVERED), self.count(FLAG)
        if covered + flags == 0:
            return False  # nothing recognised as unrevealed; cannot distinguish from misread
        if total_mines is not None:
            return covered + flags == total_mines
        return covered == 0 and flags > 0
