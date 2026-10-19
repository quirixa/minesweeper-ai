"""Fakes so the whole pipeline can be tested with no screen, browser or real mouse."""
import numpy as np

from automation.mouse_controller import MouseBackend
from capture.screen_capture import CaptureBackend, Region
from tests.simulator import Game


class FakeScreen(CaptureBackend):
    def __init__(self, game: Game, canvas=(900, 700), cell=32):
        self.game, self.canvas, self.cell = game, canvas, cell

    def desktop(self):
        return Region(0, 0, *self.canvas)

    def grab(self, region=None):
        img = self.game.render(cell=self.cell, canvas=self.canvas)
        if region:
            img = img[region.top:region.bottom, region.left:region.right]
        return img.copy()


class FakeMouse(MouseBackend):
    """Applies clicks to the simulated game; records everything."""

    def __init__(self, game: Game, cell=32):
        self.game, self.cell = game, cell
        self.log = []

    def move(self, x, y, duration):
        pass

    def click(self, x, y, button):
        self.log.append((x, y, button))
        cell = self.game.cell_at(x, y, cell=self.cell)
        if cell is None:
            return
        (self.game.reveal if button == "left" else self.game.flag)(*cell)
