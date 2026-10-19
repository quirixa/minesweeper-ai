"""Check that a click produced the visible result we expected; otherwise the agent must pause."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solver.board_state import COVERED, FLAG, MINE, BoardState


@dataclass(frozen=True)
class Action:
    kind: str                # "reveal" | "flag"
    cell: tuple[int, int]


@dataclass(frozen=True)
class VerificationResult:
    status: str              # ok | no_change | unexpected | lost | won
    detail: str

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "won")


def verify_action(before: np.ndarray, after: np.ndarray, action: Action,
                  total_mines: int | None = None) -> VerificationResult:
    if before.shape != after.shape:
        return VerificationResult("unexpected", "Board dimensions changed.")
    r, c = action.cell
    if (after == MINE).any():
        return VerificationResult("lost", "A mine is visible: the game is lost.")
    changed = before != after
    # Numbers that were already revealed must never change; flags must not vanish by themselves.
    for (i, j) in np.argwhere(changed):
        i, j = int(i), int(j)
        if (i, j) == (r, c):
            continue
        b, a = before[i, j], after[i, j]
        if b >= 0:
            return VerificationResult("unexpected", f"Revealed cell ({i+1},{j+1}) changed from {b} to {a}.")
        if b == FLAG:
            return VerificationResult("unexpected", f"Flag at ({i+1},{j+1}) disappeared.")
        if b == COVERED and a == FLAG:
            return VerificationResult("unexpected", f"A flag appeared at ({i+1},{j+1}) without being placed.")
        if b == COVERED and a >= 0 and action.kind != "reveal":
            return VerificationResult("unexpected", f"Cell ({i+1},{j+1}) was revealed by a flag action.")
    tgt_before, tgt_after = before[r, c], after[r, c]
    if action.kind == "reveal":
        if tgt_after == COVERED:
            return VerificationResult("no_change", f"Cell ({r+1},{c+1}) is still covered after clicking it.")
        if tgt_after == FLAG:
            return VerificationResult("unexpected", f"Cell ({r+1},{c+1}) became a flag after a left click.")
    elif action.kind == "flag":
        if tgt_after == COVERED:
            return VerificationResult("no_change", f"Cell ({r+1},{c+1}) was not flagged by the right click.")
        if tgt_after != FLAG:
            return VerificationResult("unexpected", f"Cell ({r+1},{c+1}) changed to {tgt_after} instead of a flag.")
    if BoardState(after).is_won(total_mines):
        return VerificationResult("won", "All safe cells are revealed.")
    return VerificationResult("ok", "The click produced the expected visible change.")
