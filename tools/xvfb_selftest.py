"""Real end-to-end check on an X11 display: MSS screen capture + PyAutoGUI clicks on a live Tk board.

    xvfb-run -a -s "-screen 0 1280x800x24" python -m tools.xvfb_selftest [seed]

Runs entirely inside the virtual display; it never touches a real desktop.
"""
from __future__ import annotations

import sys
import time

from agent import Agent, State
from automation.mouse_controller import MouseController, PyAutoGUIBackend
from capture.screen_capture import MSSBackend, Region, ScreenCapture
from config import Settings
from tools.practice_board import PracticeBoard
from vision.calibration import auto_calibrate
from vision.tile_recognition import TileRecognizer


def main(seed: int = 3) -> int:
    app = PracticeBoard(9, 9, 10, seed=seed, geometry="+80+80")
    app.update()
    time.sleep(0.5)
    app.update()
    x, y, w, h = app.winfo_rootx(), app.winfo_rooty(), app.winfo_width(), app.winfo_height()
    region = Region(x - 20, y - 40, w + 40, h + 80)
    backend = MSSBackend()
    capture = ScreenCapture(backend, fps=20)
    frame = backend.grab(region)
    cal = auto_calibrate(frame, region)
    if cal is None:
        print("FAIL: grid not detected"); return 1
    g = cal.geometry
    print(f"detected {g.rows}x{g.cols} grid, cell {g.cell_w:.1f}px, confidence {g.confidence:.0%}")
    rec = TileRecognizer()
    rec.learn_covered(frame, g)
    mouse = MouseController(PyAutoGUIBackend(), region, True, 0.05, 0.03)
    events = []
    s = Settings(manual_mode=False, approve_guesses_in_auto=False, fps=20, stable_timeout=4,
                 total_mines=10, check_active_window=False)
    agent = Agent(s, capture, mouse, cal, rec, events.append)
    agent.start()
    end = time.time() + 90
    while time.time() < end and agent.state == State.RUNNING:
        app.update()
        time.sleep(0.02)
    final = agent.state
    agent.stop(); agent.join(3)
    print("final state:", final.value, "| game won:", app.game.won, "lost:", app.game.lost,
          "| clicks:", mouse.click_count, "| guesses:", agent.stats.guesses)
    for e in [e for e in events if e.kind in ("log", "reasoning")][-4:]:
        print("  ", e.text[:140])
    app.destroy()
    return 0 if (app.game.won or app.game.lost) else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
