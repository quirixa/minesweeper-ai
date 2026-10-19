"""Deterministic solver: simple rules -> subset rule -> exhaustive enumeration -> best guess.

Every Move carries an explanation generated from the exact numbers/constraints that produced it.
`certain=True` is only used for conclusions that hold in every layout consistent with the board.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .board_state import COVERED, BoardState, Cell
from .constraints import Constraint, ContradictionError, build_constraints
from .probability_solver import ProbabilityResult, TooComplexError, compute_probabilities


@dataclass
class Move:
    kind: str                      # "reveal" or "flag"
    cell: Cell
    certain: bool
    mine_probability: float | None
    explanation: str
    rule: str                      # basic | subset | enumeration | guess | opening

    @property
    def is_guess(self) -> bool:
        return not self.certain


@dataclass
class Analysis:
    safe_moves: list[Move] = field(default_factory=list)
    mine_moves: list[Move] = field(default_factory=list)
    guess: Move | None = None
    messages: list[str] = field(default_factory=list)
    contradiction: str | None = None
    probabilities: dict = field(default_factory=dict)
    enumeration_skipped: bool = False

    @property
    def has_certain_move(self) -> bool:
        return bool(self.safe_moves or self.mine_moves)


def _pos(cell: Cell) -> str:
    return f"row {cell[0]+1}, column {cell[1]+1}"


class LogicSolver:
    def __init__(self, total_mines: int | None = None, node_limit: int = 3_000_000):
        self.total_mines = total_mines
        self.node_limit = node_limit

    def analyze(self, board: BoardState) -> Analysis:
        a = Analysis()
        a.messages.append("Analyzing the board...")
        covered = board.covered_cells()
        if not covered:
            a.messages.append("There are no covered cells left to analyze.")
            return a
        if board.revealed_count == 0 and board.flag_count == 0:
            return self._opening(board, a)
        try:
            cons = build_constraints(board)
        except ContradictionError as e:
            a.contradiction = str(e)
            a.messages.append(f"Contradiction: {e} The board was probably misread; refusing to move.")
            return a

        safe: dict[Cell, Move] = {}
        mines: dict[Cell, Move] = {}
        self._basic_rules(cons, safe, mines)
        self._subset_rule(cons, safe, mines)

        prob: ProbabilityResult | None = None
        try:
            prob = compute_probabilities(board, self.total_mines, cons, self.node_limit)
        except ContradictionError as e:
            a.contradiction = str(e)
            a.messages.append(f"Contradiction: {e} Refusing to move.")
            return a
        except TooComplexError:
            a.enumeration_skipped = True
            a.messages.append("The frontier is too large to enumerate exactly; using simple rules "
                              "and an approximate local estimate only.")
        if prob:
            a.probabilities = dict(prob.probabilities)
            self._enumeration_moves(prob, safe, mines)

        a.safe_moves = sorted(safe.values(), key=lambda m: m.cell)
        a.mine_moves = sorted(mines.values(), key=lambda m: m.cell)
        if a.safe_moves:
            a.messages.append(f"{len(a.safe_moves)} guaranteed-safe cell(s) and "
                              f"{len(a.mine_moves)} proven mine(s) found.")
        elif a.mine_moves:
            a.messages.append(f"{len(a.mine_moves)} proven mine(s) found, but no new safe cell yet.")
        if not a.safe_moves:
            self._choose_guess(board, a, prob, cons)
        return a

    # ---- rule 1: single number ---------------------------------------------------------
    def _basic_rules(self, cons, safe, mines):
        for con in cons:
            r, c = con.origin
            fl = (f" and {con.flagged} flagged neighbour(s)" if con.flagged else " and no flagged neighbours")
            if con.mines == 0:
                for p in con.cells:
                    safe.setdefault(p, Move(
                        "reveal", p, True, 0.0,
                        f"This cell is safe because the {con.number} at {_pos(con.origin)} already has "
                        f"{con.flagged} confirmed neighbouring mine(s), which accounts for its whole count.",
                        "basic"))
            elif con.mines == len(con.cells):
                for p in con.cells:
                    mines.setdefault(p, Move(
                        "flag", p, True, 1.0,
                        f"The {con.number} at {_pos(con.origin)} has {len(con.cells)} unrevealed "
                        f"neighbour(s){fl}; all of them must contain a mine.",
                        "basic"))

    # ---- rule 2: subset rule -----------------------------------------------------------
    def _subset_rule(self, cons: list[Constraint], safe, mines):
        by_cell: dict[Cell, list[Constraint]] = {}
        for con in cons:
            for p in con.cells:
                by_cell.setdefault(p, []).append(con)
        seen = set()
        for a in cons:
            peers = {id(b): b for p in a.cells for b in by_cell[p]}
            for b in peers.values():
                if a is b or (id(a), id(b)) in seen:
                    continue
                seen.add((id(a), id(b)))
                if a.cells < b.cells:  # strict subset
                    diff = b.cells - a.cells
                    dm = b.mines - a.mines
                    why = (f"The {a.number} at {_pos(a.origin)} touches a subset of the cells around the "
                           f"{b.number} at {_pos(b.origin)}; the extra {len(diff)} cell(s) hold "
                           f"{b.mines} - {a.mines} = {dm} mine(s).")
                    if dm == 0:
                        for p in diff:
                            safe.setdefault(p, Move("reveal", p, True, 0.0, why + " So this cell is safe.", "subset"))
                    elif dm == len(diff):
                        for p in diff:
                            mines.setdefault(p, Move("flag", p, True, 1.0, why + " So this cell is a mine.", "subset"))

    # ---- rule 3: exhaustive enumeration -------------------------------------------------
    def _enumeration_moves(self, prob: ProbabilityResult, safe, mines):
        how = ("with the known total mine count" if prob.exact_total_used else "from the visible numbers")
        n = prob.configurations
        for p in sorted(prob.certain_safe):
            safe.setdefault(p, Move(
                "reveal", p, True, 0.0,
                f"Checking every mine layout consistent with the board {how}, this cell is safe in "
                f"all of them ({n} weighted layouts).", "enumeration"))
        for p in sorted(prob.certain_mine):
            mines.setdefault(p, Move(
                "flag", p, True, 1.0,
                f"Checking every mine layout consistent with the board {how}, this cell is a mine "
                f"in all of them ({n} weighted layouts).", "enumeration"))

    # ---- guessing -----------------------------------------------------------------------
    def _opening(self, board: BoardState, a: Analysis) -> Analysis:
        cell = (board.rows // 2, board.cols // 2)
        p = None
        if self.total_mines is not None:
            p = self.total_mines / (board.rows * board.cols)
        a.guess = Move("reveal", cell, False, p,
                       "The board is untouched, so no information exists yet. This is a GUESS: "
                       "opening in the middle gives the best chance of a large cascade."
                       + (f" Prior mine probability: {p:.0%}." if p is not None else ""), "opening")
        a.messages.append("No guaranteed safe move is available (nothing revealed yet).")
        return a

    def _choose_guess(self, board, a: Analysis, prob, cons):
        if prob is None:  # enumeration failed: crude local estimate
            est: dict[Cell, float] = {}
            for con in cons:
                for p in con.cells:
                    est[p] = max(est.get(p, 0.0), con.mines / len(con.cells))
            if not est:
                a.messages.append("Recognition confidence is fine but information is insufficient.")
                return
            cell = min(est, key=lambda p: (est[p], p))
            a.guess = Move("reveal", cell, False, est[cell],
                           f"GUESS (approximate): no exact analysis possible; this cell has the lowest "
                           f"local risk estimate, about {est[cell]:.0%}.", "guess")
            return
        probs = prob.probabilities
        if not probs:
            a.messages.append("The visible information is insufficient to rank any cell.")
            return
        # Lowest risk first; among ties prefer frontier cells (they reveal information) then position.
        cell = min(probs, key=lambda p: (round(probs[p], 12), p not in prob.frontier, p))
        shown = sorted(set(round(v, 2) for k, v in probs.items() if k in prob.frontier))
        if shown:
            a.messages.append(f"{len(prob.frontier)} frontier cell(s) remain uncertain. Estimated mine "
                              f"probabilities: " + ", ".join(f"{v:.0%}" for v in shown[:6])
                              + ("..." if len(shown) > 6 else "") + ".")
        a.messages.append("No guaranteed safe move is available. The AI is selecting the lowest-risk candidate.")
        where = "frontier cell" if cell in prob.frontier else "cell away from the numbers"
        est = " (an estimate: total mine count unknown)" if (cell not in prob.frontier and prob.other_is_estimate) else ""
        a.guess = Move("reveal", cell, False, probs[cell],
                       f"GUESS, not a certainty: this {where} has about a {probs[cell]:.0%} chance of being "
                       f"a mine{est}, the lowest available. Probabilities come from exact counting of "
                       f"{prob.configurations} layouts.", "guess")
