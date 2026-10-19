import pytest
from solver.board_state import BoardState
from solver.constraints import ContradictionError, build_constraints, split_components


def test_constraint_subtracts_flags():
    cons = build_constraints(BoardState.from_rows(["F2#", "11#"]))
    con = next(c for c in cons if c.origin == (0, 1))
    assert con.number == 2 and con.flagged == 1 and con.mines == 1 and con.cells == {(0, 2), (1, 2)}


def test_too_many_flags_raises():
    with pytest.raises(ContradictionError):
        build_constraints(BoardState.from_rows(["F1F", "11#"]))


def test_not_enough_covered_raises():
    with pytest.raises(ContradictionError):
        build_constraints(BoardState.from_rows(["3#", "11"]))


def test_components_are_independent():
    comps = split_components(build_constraints(BoardState.from_rows(["#1...1#"])))
    assert len(comps) == 2
