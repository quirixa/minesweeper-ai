"""A local Tk Minesweeper to practise on (left click = reveal, right click = flag).

    python -m tools.practice_board [rows cols mines]

It exists so you can calibrate and test the AI without needing any website. It shares game logic
with the test simulator (tests/simulator.py).
"""
from __future__ import annotations

import sys
import tkinter as tk

from tests.simulator import COLORS, Game

CELL = 32


class PracticeBoard(tk.Tk):
    def __init__(self, rows=9, cols=9, mines=10, seed=None, geometry="+60+60"):
        super().__init__()
        self.title("Practice Minesweeper")
        self.rows, self.cols, self.mines, self.seed = rows, cols, mines, seed
        self.geometry(geometry)
        self.status = tk.Label(self, text="", font=("TkDefaultFont", 11))
        self.status.pack(fill="x")
        self.canvas = tk.Canvas(self, width=cols * CELL, height=rows * CELL, highlightthickness=0, bg="#f5f5f5")
        self.canvas.pack(padx=20, pady=20)
        self.canvas.bind("<Button-1>", lambda e: self._click(e, "reveal"))
        self.canvas.bind("<Button-3>", lambda e: self._click(e, "flag"))
        tk.Button(self, text="New game", command=self.new_game).pack(pady=(0, 10))
        self.new_game()

    def new_game(self):
        self.game = Game(self.rows, self.cols, self.mines, self.seed)
        self.draw()

    def _click(self, e, kind):
        r, c = e.y // CELL, e.x // CELL
        if 0 <= r < self.rows and 0 <= c < self.cols:
            (self.game.reveal if kind == "reveal" else self.game.flag)(r, c)
            self.draw()

    def draw(self):
        g, cv = self.game, self.canvas
        cv.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c * CELL, r * CELL
                shown = g.revealed[r, c] or (g.lost and (r, c) in g.mines)
                if not shown:
                    cv.create_rectangle(x, y, x + CELL, y + CELL, fill="#bebebe", outline="")
                    cv.create_polygon(x, y, x + CELL, y, x + CELL - 3, y + 3, x + 3, y + 3, x + 3, y + CELL - 3, x, y + CELL, fill="white", outline="")
                    cv.create_polygon(x + CELL, y + CELL, x, y + CELL, x + 3, y + CELL - 3, x + CELL - 3, y + CELL - 3, x + CELL - 3, y + 3, x + CELL, y, fill="#6e6e6e", outline="")
                    if g.flags[r, c]:
                        cv.create_polygon(x + 10, y + 8, x + 24, y + 13, x + 10, y + 18, fill="#dc0000", outline="")
                        cv.create_line(x + 10, y + 8, x + 10, y + 26, width=2, fill="black")
                else:
                    cv.create_rectangle(x, y, x + CELL, y + CELL, fill="#a5a5a5", outline="#787878")
                    if (r, c) in g.mines:
                        cv.create_oval(x + 6, y + 6, x + CELL - 6, y + CELL - 6, fill="black")
                    elif g.number(r, c):
                        n = g.number(r, c)
                        b, gr, rd = COLORS[n]
                        cv.create_text(x + CELL // 2, y + CELL // 2, text=str(n), fill="#%02x%02x%02x" % (rd, gr, b),
                                       font=("TkDefaultFont", 16, "bold"))
        self.status.configure(text="YOU LOST" if g.lost else "YOU WON" if g.won else "Playing")


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:4]]
    PracticeBoard(*(a or [9, 9, 10])).mainloop()
