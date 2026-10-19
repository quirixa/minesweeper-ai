"""Builds the real GUI on a virtual display (skipped if no display) and exercises the event pump."""
import os
import numpy as np
import pytest

pytest.importorskip("tkinter")
pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs an X display (use xvfb-run)")


def test_gui_builds_and_handles_events():
    from agent import Event
    from interface.app import App
    app = App()
    app.update()
    assert str(app.btn_pause.cget("text")) == "Pause"
    app.events.put(Event("reasoning", "The number 1 has one unrevealed neighbor."))
    app.events.put(Event("state", "Running"))
    app.events.put(Event("stats", "", dict(revealed=5, flagged=1, guesses=2, wins=0, losses=1, clicks=3)))
    app.events.put(Event("board", "", dict(grid=np.zeros((3, 3), int), confidence=np.ones((3, 3)))))
    app.events.put(Event("history", "", dict(cell=(0, 0), action="reveal", certain=True, probability=0.0, outcome="ok")))
    app._poll(); app.update()
    assert app.stat_vars["Guesses"].get() == "2" and app.stat_vars["Board"].get() == "3 x 3"
    assert len(app.reasoning.tree.get_children()) == 1
    app.emergency()
    assert app.mouse.stopped
    app._close()
