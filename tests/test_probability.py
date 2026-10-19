import itertools
import random

import numpy as np
import pytest

from solver.board_state import COVERED, BoardState
from solver.constraints import ContradictionError
from solver.probability_solver import compute_probabilities


def test_fifty_fifty():
    p = compute_probabilities(BoardState.from_rows(["#1", "#1"]))
    assert p.probabilities[(0, 0)] == pytest.approx(0.5) and p.probabilities[(1, 0)] == pytest.approx(0.5)
    assert not p.certain_safe and not p.certain_mine


def test_two_mines_in_three_cells():
    p = compute_probabilities(BoardState.from_rows(["##", "2#"]))
    assert all(v == pytest.approx(2 / 3) for v in p.probabilities.values())


def test_total_count_weights_unconstrained_cells():
    b = BoardState.from_rows(["#1##", "#1##"])
    p = compute_probabilities(b, total_mines=2)
    # 1 mine in the frontier (4 cells), 1 in the 2 far cells
    assert p.probabilities[(0, 3)] == pytest.approx(0.5)
    assert sum(p.probabilities.values()) == pytest.approx(2.0)
    assert p.exact_total_used


def test_inconsistent_total_raises():
    with pytest.raises(ContradictionError):
        compute_probabilities(BoardState.from_rows(["#1", "11"]), total_mines=5)


def test_unknown_total_marks_other_probability_as_estimate():
    p = compute_probabilities(BoardState.from_rows(["#1##", "#1##"]))
    assert p.other_is_estimate


def _random_board(rng, rows=4, cols=4, mines=4, reveal=0.5):
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    mset = set(rng.sample(cells, mines))
    grid = np.full((rows, cols), COVERED, dtype=int)
    b = BoardState(grid)
    for (r, c) in cells:
        if (r, c) not in mset and rng.random() < reveal:
            grid[r, c] = sum(1 for q in b.neighbors(r, c) if q in mset)
    return b


def _brute(b, total):
    cov = b.covered_cells()
    layouts = []
    for bits in itertools.product((0, 1), repeat=len(cov)):
        if total is not None and sum(bits) != total:
            continue
        m = {c for c, v in zip(cov, bits) if v}
        if all(sum(q in m for q in b.neighbors(r, c)) == b.grid[r, c]
               for r in range(b.rows) for c in range(b.cols) if b.grid[r, c] >= 0):
            layouts.append(m)
    return cov, layouts


@pytest.mark.parametrize("use_total", [False, True])
def test_matches_brute_force_on_random_boards(use_total):
    rng = random.Random(7)
    checked = 0
    for _ in range(60):
        b = _random_board(rng)
        if len(b.covered_cells()) > 14 or b.revealed_count == 0:
            continue
        total = 4 if use_total else None
        cov, layouts = _brute(b, total)
        p = compute_probabilities(b, total)
        for cell in cov:
            if not use_total and cell not in p.frontier:
                continue
            expected = sum(cell in m for m in layouts) / len(layouts)
            assert p.probabilities[cell] == pytest.approx(expected), (b.grid, cell)
        checked += 1
    assert checked >= 10
