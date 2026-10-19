"""Exact mine probabilities by enumerating every layout consistent with the visible numbers.

Frontier cells (covered cells touching a number) split into independent components. Each
component is enumerated by backtracking with constraint pruning, giving, per mine count k,
the number of valid layouts and how many of them place a mine on each cell. Components are
then combined; if the total mine count is known, each combination is weighted by the number
of ways to place the leftover mines on the non-frontier ("other") cells: C(M, R - s).
All counting uses Python integers, so results are exact (no floating-point drift).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

from .board_state import COVERED, FLAG, BoardState, Cell
from .constraints import Constraint, ContradictionError, build_constraints, split_components


class TooComplexError(Exception):
    """Enumeration exceeded its node budget; fall back to rule-based reasoning."""


@dataclass
class ProbabilityResult:
    probabilities: dict[Cell, float] = field(default_factory=dict)  # frontier + other cells
    certain_safe: set = field(default_factory=set)
    certain_mine: set = field(default_factory=set)
    frontier: set = field(default_factory=set)
    other_cells: list = field(default_factory=list)
    other_probability: float | None = None
    other_is_estimate: bool = False
    configurations: int = 0     # layouts counted (product over components, or weighted total)
    exact_total_used: bool = False


def enumerate_component(cells: list[Cell], cons: list[Constraint], node_limit: int = 3_000_000):
    """Return {k: [layout_count, per_cell_mine_counts]} for one connected component."""
    idx = {c: i for i, c in enumerate(cells)}
    c_cells = [[idx[p] for p in con.cells] for con in cons]
    cell_cons: list[list[int]] = [[] for _ in cells]
    for ci, members in enumerate(c_cells):
        for i in members:
            cell_cons[i].append(ci)
    # Order cells by BFS so constraints close early and prune hard.
    order, seen = [], set()
    for start in range(len(cells)):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        while queue:
            i = queue.pop(0)
            order.append(i)
            for ci in cell_cons[i]:
                for j in c_cells[ci]:
                    if j not in seen:
                        seen.add(j)
                        queue.append(j)
    remaining = [con.mines for con in cons]
    unassigned = [len(m) for m in c_cells]
    assign = [0] * len(cells)
    stats: dict[int, list] = {}
    nodes = 0

    def rec(pos: int, mines: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise TooComplexError
        if pos == len(order):
            entry = stats.setdefault(mines, [0, [0] * len(cells)])
            entry[0] += 1
            for j, v in enumerate(assign):
                if v:
                    entry[1][j] += 1
            return
        i = order[pos]
        for v in (0, 1):
            ok = True
            for ci in cell_cons[i]:
                unassigned[ci] -= 1
                remaining[ci] -= v
            for ci in cell_cons[i]:
                if remaining[ci] < 0 or remaining[ci] > unassigned[ci]:
                    ok = False
                    break
            if ok:
                assign[i] = v
                rec(pos + 1, mines + v)
            assign[i] = 0
            for ci in cell_cons[i]:
                unassigned[ci] += 1
                remaining[ci] += v

    rec(0, 0)
    return stats


def _convolve(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, x in a.items():
        for j, y in b.items():
            out[i + j] = out.get(i + j, 0) + x * y
    return out


def compute_probabilities(board: BoardState, total_mines: int | None = None,
                          constraints: list[Constraint] | None = None,
                          node_limit: int = 3_000_000) -> ProbabilityResult:
    """Raises ContradictionError (impossible board) or TooComplexError (too large to enumerate)."""
    constraints = build_constraints(board) if constraints is None else constraints
    comps = split_components(constraints)
    frontier = {p for cells, _ in comps for p in cells}
    others = [p for p in board.covered_cells() if p not in frontier]
    res = ProbabilityResult(frontier=frontier, other_cells=others)

    comp_stats = []
    for cells, cons in comps:
        st = enumerate_component(cells, cons, node_limit)
        if not st:
            raise ContradictionError("No mine layout satisfies the numbers around "
                                     f"row {cons[0].origin[0]+1}, column {cons[0].origin[1]+1}.")
        comp_stats.append((cells, st))

    counts_per_comp = [{k: v[0] for k, v in st.items()} for _, st in comp_stats]
    M = len(others)
    R = None if total_mines is None else total_mines - board.flag_count

    def others_weight(s: int) -> int:
        if R is None:
            return 1
        left = R - s
        return comb(M, left) if 0 <= left <= M else 0

    # total layouts weight and expected mines in "other" cells
    total_conv: dict[int, int] = {0: 1}
    for cc in counts_per_comp:
        total_conv = _convolve(total_conv, cc)
    if R is None:
        den_all = 1
        for cc in counts_per_comp:
            den_all *= sum(cc.values())
        res.configurations = den_all
    else:
        den_all = sum(w * others_weight(s) for s, w in total_conv.items())
        if den_all == 0:
            raise ContradictionError(
                f"The configured total of {total_mines} mines is inconsistent with the visible board.")
        res.configurations = den_all
        res.exact_total_used = True

    # per-cell frontier probabilities
    for ci, (cells, st) in enumerate(comp_stats):
        rest = {0: 1}
        for cj, cc in enumerate(counts_per_comp):
            if cj != ci:
                rest = _convolve(rest, cc)
        for j, cell in enumerate(cells):
            num = 0
            for k, (_, per_cell) in st.items():
                if R is None:
                    comp_total = sum(counts_per_comp[ci].values())
                    num += per_cell[j] * 1
                    den = comp_total
                else:
                    ways = sum(w * others_weight(k + s) for s, w in rest.items())
                    num += per_cell[j] * ways
                    den = den_all
            res.probabilities[cell] = num / den
            if num == 0:
                res.certain_safe.add(cell)
            elif num == den:
                res.certain_mine.add(cell)

    # non-frontier cells
    if others:
        if R is None:
            res.other_is_estimate = True
            vals = list(res.probabilities.values())
            res.other_probability = sum(vals) / len(vals) if vals else None
        else:
            exp_num = sum(w * others_weight(s) * (R - s) for s, w in total_conv.items()
                          if others_weight(s))
            p = exp_num / (M * den_all)
            res.other_probability = p
            for cell in others:
                res.probabilities[cell] = p
            if exp_num == 0:
                res.certain_safe.update(others)
            elif exp_num == M * den_all:
                res.certain_mine.update(others)
    return res
