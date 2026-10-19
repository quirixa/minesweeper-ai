"""End-to-end: simulated game -> screen capture -> vision -> solver -> (fake) mouse -> verification."""
import time

import pytest

from agent import Agent, State
from automation.mouse_controller import MouseController
from capture.screen_capture import Region, ScreenCapture
from config import Settings
from tests.fakes import FakeMouse, FakeScreen
from tests.simulator import Game
from vision.calibration import auto_calibrate
from vision.tile_recognition import TileRecognizer


def build(seed, rows=9, cols=9, mines=10, **settings):
    game = Game(rows, cols, mines, seed)
    screen = FakeScreen(game)
    capture = ScreenCapture(screen, fps=200)
    fresh = screen.grab()
    region = Region(0, 0, 900, 700)
    calib = auto_calibrate(fresh, region)
    assert calib and (calib.geometry.rows, calib.geometry.cols) == (rows, cols)
    rec = TileRecognizer()
    rec.learn_covered(fresh, calib.geometry)
    fm = FakeMouse(game)
    mouse = MouseController(fm, region, True, 0, 0)
    s = Settings(manual_mode=False, approve_guesses_in_auto=False, fps=200, stable_timeout=3,
                 total_mines=mines, check_active_window=False, **settings)
    events = []
    agent = Agent(s, capture, mouse, calib, rec, events.append)
    return game, agent, fm, events


def run(agent, timeout=60):
    """Run until the agent finishes or pauses, then shut the worker down."""
    agent.start()
    end = time.time() + timeout
    while time.time() < end and agent.state == State.RUNNING:
        time.sleep(0.02)
    assert agent.state != State.RUNNING, "agent did not finish or pause in time"
    final = agent.state
    agent.stop()
    agent.join(5)
    return final


@pytest.mark.parametrize("seed", range(6))
def test_full_game_certain_moves_are_always_correct(seed):
    game, agent, fm, events = build(seed)
    assert run(agent) in (State.STOPPED, State.PAUSED)
    props = [e for e in events if e.kind == "proposal"]
    assert props, "agent never proposed a move"
    # Replay: every click the agent flagged as *certain* must have been correct in the real game.
    for e in events:
        if e.kind == "history" and e.data["certain"]:
            r, c = e.data["cell"]
            if e.data["action"] == "reveal":
                assert (r, c) not in game.mines, "a 'certain safe' cell was a mine!"
            else:
                assert (r, c) in game.mines, "a 'certain mine' flag was wrong!"
    # the final outcome is reported coherently
    if game.won:
        assert agent.stats.wins == 1
    if game.lost:
        assert agent.stats.losses == 1 and any("lost" in e.text.lower() for e in events if e.kind == "log")


def test_at_least_one_simulated_game_is_won():
    wins = 0
    for seed in range(6):
        game, agent, fm, events = build(seed)
        run(agent)
        wins += game.won
    assert wins >= 1


def test_manual_mode_waits_for_approval_and_does_not_click():
    game, agent, fm, events = build(1)
    agent.s.manual_mode = True
    agent.start()
    deadline = time.time() + 10
    while time.time() < deadline and not any(e.kind == "proposal" and e.data["awaiting"] for e in events):
        time.sleep(0.05)
    assert any(e.kind == "proposal" and e.data["awaiting"] for e in events)
    time.sleep(0.3)
    assert fm.log == [], "clicked without approval"
    agent.approve()
    deadline = time.time() + 10
    while time.time() < deadline and not fm.log:
        time.sleep(0.05)
    assert fm.log, "did not click after approval"
    agent.emergency_stop()
    agent.join(5)


def test_emergency_stop_prevents_any_further_click():
    game, agent, fm, events = build(2)
    agent.s.manual_mode = True
    agent.start()
    time.sleep(0.5)
    agent.emergency_stop()
    agent.approve()
    agent.join(5)
    assert fm.log == [] and agent.state == State.STOPPED and agent.mouse.stopped


def test_low_confidence_pauses_instead_of_clicking():
    game, agent, fm, events = build(3, min_confidence=1.01)  # impossible threshold
    agent.s.stable_timeout = 1.0
    assert run(agent, 20) == State.PAUSED and fm.log == []
    assert any("confidence is too low" in e.text for e in events if e.kind == "reasoning")


def test_guess_requires_approval_even_in_auto_mode():
    game, agent, fm, events = build(4)
    agent.s.approve_guesses_in_auto = True
    agent.start()
    deadline = time.time() + 10
    while time.time() < deadline and not any(e.kind == "proposal" and e.data["awaiting"] for e in events):
        time.sleep(0.05)
    first = next(e for e in events if e.kind == "proposal")
    assert first.data["certain"] is False and first.data["awaiting"]  # the opening click is a guess
    assert fm.log == []
    agent.emergency_stop()
    agent.join(5)
