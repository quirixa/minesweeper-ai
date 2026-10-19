import numpy as np
import pytest

from automation.action_verification import Action, verify_action
from automation.mouse_controller import EmergencyStop, MouseBackend, MouseController, OutsideRegionError
from capture.screen_capture import Region
from solver.board_state import COVERED, FLAG, MINE


class Rec(MouseBackend):
    def __init__(self):
        self.moves, self.clicks = [], []

    def move(self, x, y, d):
        self.moves.append((x, y))

    def click(self, x, y, b):
        self.clicks.append((x, y, b))


def ctl(**kw):
    b = Rec()
    return b, MouseController(b, Region(100, 100, 200, 200), move_duration=0, click_delay=0, **kw)


def test_click_inside_region_works():
    b, m = ctl()
    m.click(150, 150, "left")
    assert b.clicks == [(150, 150, "left")] and m.click_count == 1


def test_click_outside_region_refused():
    b, m = ctl()
    with pytest.raises(OutsideRegionError):
        m.click(10, 10)
    assert not b.clicks


def test_emergency_stop_blocks_all_further_input():
    b, m = ctl()
    m.emergency_stop()
    with pytest.raises(EmergencyStop):
        m.click(150, 150)
    assert not b.clicks and not b.moves and m.stopped
    m.reset_emergency_stop()
    m.click(150, 150)
    assert len(b.clicks) == 1


def test_failsafe_exception_from_backend_becomes_emergency_stop():
    class Boom(Rec):
        def click(self, x, y, b):
            raise EmergencyStop("corner")
    m = MouseController(Boom(), None, restrict_to_region=False, move_duration=0, click_delay=0)
    with pytest.raises(EmergencyStop):
        m.click(5, 5)
    assert m.stopped


def test_pyautogui_failsafe_is_enabled_in_backend_source():
    import inspect
    from automation import mouse_controller
    assert "FAILSAFE = True" in inspect.getsource(mouse_controller.PyAutoGUIBackend)


def grid(rows):
    m = {"#": COVERED, "F": FLAG, "*": MINE, ".": 0}
    return np.array([[m.get(ch, None) if ch in m else int(ch) for ch in r] for r in rows])


def test_verify_expected_reveal():
    before, after = grid(["##", "##"]), grid(["1#", "##"])
    assert verify_action(before, after, Action("reveal", (0, 0))).status == "ok"


def test_verify_no_change_detected():
    g = grid(["##", "##"])
    assert verify_action(g, g.copy(), Action("reveal", (0, 0))).status == "no_change"


def test_verify_unexpected_change_and_stale_numbers():
    before = grid(["1#", "##"])
    after = grid(["2#", "1#"])
    assert verify_action(before, after, Action("reveal", (1, 0))).status == "unexpected"


def test_verify_flag_and_loss():
    before = grid(["##", "##"])
    assert verify_action(before, grid(["F#", "##"]), Action("flag", (0, 0))).status == "ok"
    assert verify_action(before, before, Action("flag", (0, 0))).status == "no_change"
    assert verify_action(before, grid(["*#", "##"]), Action("reveal", (0, 0))).status == "lost"


def test_unrequested_flag_is_unexpected():
    assert verify_action(grid(["##", "##"]), grid(["1F", "##"]), Action("reveal", (0, 0))).status == "unexpected"
