"""Turn visible numbers into constraints: 'exactly k mines among this set of covered cells'."""
from __future__ import annotations

from dataclasses import dataclass

from .board_state import COVERED, FLAG, BoardState, Cell


class ContradictionError(Exception):
    """The visible board cannot be produced by any mine layout (misread or wrong flag)."""


@dataclass(frozen=True)
class Constraint:
    origin: Cell
    number: int
    flagged: int
    cells: frozenset  # covered neighbours
    mines: int        # mines still hidden among `cells` (= number - flagged)


def build_constraints(board: BoardState) -> list[Constraint]:
    out: list[Constraint] = []
    for r in range(board.rows):
        for c in range(board.cols):
            n = int(board.grid[r, c])
            if n < 0:
                continue
            nb = board.neighbors(r, c)
            covered = frozenset(p for p in nb if board.grid[p] == COVERED)
            flagged = sum(1 for p in nb if board.grid[p] == FLAG)
            remaining = n - flagged
            if remaining < 0:
                raise ContradictionError(
                    f"The {n} at row {r+1}, column {c+1} has {flagged} flagged neighbours (more than {n}).")
            if remaining > len(covered):
                raise ContradictionError(
                    f"The {n} at row {r+1}, column {c+1} needs {remaining} more mine(s) but only "
                    f"{len(covered)} covered neighbour(s) remain.")
            if covered:
                out.append(Constraint((r, c), n, flagged, covered, remaining))
    return out


def split_components(constraints: list[Constraint]) -> list[tuple[list[Cell], list[Constraint]]]:
    """Group constraints that share cells (union-find); each group can be solved independently."""
    parent: dict[Cell, Cell] = {}

    def find(x: Cell) -> Cell:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for con in constraints:
        cells = list(con.cells)
        for p in cells:
            parent.setdefault(p, p)
        for p in cells[1:]:
            parent[find(p)] = find(cells[0])
    groups: dict[Cell, tuple[set, list]] = {}
    for con in constraints:
        root = find(next(iter(con.cells)))
        g = groups.setdefault(root, (set(), []))
        g[0].update(con.cells)
        g[1].append(con)
    return [(sorted(cells), cons) for cells, cons in groups.values()]
