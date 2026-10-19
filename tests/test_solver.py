from solver.board_state import BoardState
from solver.logic_solver import LogicSolver


def cells(moves):
    return {m.cell for m in moves}


def test_one_with_single_covered_neighbor_is_mine():
    a = LogicSolver().analyze(BoardState.from_rows(["1#", "11"]))
    assert cells(a.mine_moves) == {(0, 1)}
    m = a.mine_moves[0]
    assert m.certain and "must contain a mine" in m.explanation and m.mine_probability == 1.0


def test_flagged_neighbor_makes_other_cells_safe():
    a = LogicSolver().analyze(BoardState.from_rows(["F1#", "11#"]))
    assert cells(a.safe_moves) == {(0, 2), (1, 2)}
    assert all(m.certain for m in a.safe_moves)
    assert "already has 1 confirmed" in a.safe_moves[0].explanation


def test_121_pattern():
    a = LogicSolver().analyze(BoardState.from_rows(["###", "121", "..."]))
    assert cells(a.mine_moves) == {(0, 0), (0, 2)} and cells(a.safe_moves) == {(0, 1)}


def test_multiple_overlapping_constraints():
    # a+b=1, a+b+c=1, b+c=1  ->  c=0, b=1, a=0
    a = LogicSolver().analyze(BoardState.from_rows(["###", "111"]))
    assert cells(a.mine_moves) == {(0, 1)} and cells(a.safe_moves) == {(0, 0), (0, 2)}


def test_contradiction_too_many_flags_blocks_all_moves():
    a = LogicSolver().analyze(BoardState.from_rows(["1F#", "FF#"]))
    assert a.contradiction and not a.has_certain_move and a.guess is None


def test_contradiction_impossible_number():
    assert LogicSolver().analyze(BoardState.from_rows(["8#", "##"])).contradiction


def test_guess_is_labelled_uncertain():
    a = LogicSolver().analyze(BoardState.from_rows(["#1#", "#1#"]))
    assert not a.has_certain_move
    g = a.guess
    assert g and g.certain is False and g.is_guess
    assert abs(g.mine_probability - 0.25) < 1e-9
    assert "GUESS" in g.explanation and "safe" not in g.explanation.lower().replace("lowest", "")


def test_certain_move_is_not_a_guess():
    a = LogicSolver().analyze(BoardState.from_rows(["###", "121", "..."]))
    assert all(not m.is_guess for m in a.safe_moves + a.mine_moves)
    assert a.guess is None


def test_opening_is_a_guess():
    a = LogicSolver(total_mines=3).analyze(BoardState.from_rows(["####"] * 4))
    assert a.guess.rule == "opening" and not a.guess.certain and not a.safe_moves


def test_total_mine_count_proves_far_cells_safe():
    b = BoardState.from_rows(["#1##", "#1##"])
    assert not LogicSolver().analyze(b).safe_moves
    a = LogicSolver(total_mines=1).analyze(b)
    assert cells(a.safe_moves) == {(0, 3), (1, 3)}
    assert all(m.rule == "enumeration" for m in a.safe_moves)
    a3 = LogicSolver(total_mines=3).analyze(b)
    assert cells(a3.mine_moves) == {(0, 3), (1, 3)}


def test_won_lost_detection():
    assert BoardState.from_rows(["1*", "11"]).is_lost
    assert BoardState.from_rows(["1F", "11"]).is_won(total_mines=1)
    assert not BoardState.from_rows(["1#", "11"]).is_won(total_mines=2)
