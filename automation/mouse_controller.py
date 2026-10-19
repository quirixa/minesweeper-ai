"""Real mouse control with an emergency stop, region restriction and PyAutoGUI's corner failsafe."""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

from capture.screen_capture import Region


class EmergencyStop(Exception):
    """Raised when automated input was disabled (button, failsafe corner, or program request)."""


class OutsideRegionError(Exception):
    """A click was requested outside the calibrated game region."""


class MouseBackend(ABC):
    @abstractmethod
    def move(self, x: int, y: int, duration: float) -> None: ...

    @abstractmethod
    def click(self, x: int, y: int, button: str) -> None: ...


class PyAutoGUIBackend(MouseBackend):
    def __init__(self) -> None:
        import pyautogui
        pyautogui.FAILSAFE = True   # slam the pointer into a screen corner to abort. Never disable.
        pyautogui.PAUSE = 0
        self._pg = pyautogui

    def move(self, x, y, duration):
        self._call(self._pg.moveTo, x, y, duration=duration)

    def click(self, x, y, button):
        self._call(self._pg.click, x, y, button=button)

    def _call(self, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except self._pg.FailSafeException as e:
            raise EmergencyStop("PyAutoGUI corner failsafe triggered") from e


class MouseController:
    def __init__(self, backend: MouseBackend, region: Region | None = None, restrict_to_region: bool = True,
                 move_duration: float = 0.25, click_delay: float = 0.15):
        self.backend = backend
        self.region = region
        self.restrict = restrict_to_region
        self.move_duration = move_duration
        self.click_delay = click_delay
        self._estop = threading.Event()
        self.click_count = 0

    # ---- emergency stop -------------------------------------------------------------------
    def emergency_stop(self) -> None:
        self._estop.set()

    def reset_emergency_stop(self) -> None:
        self._estop.clear()

    @property
    def stopped(self) -> bool:
        return self._estop.is_set()

    def _check(self) -> None:
        if self._estop.is_set():
            raise EmergencyStop("Emergency stop is active; automated input is disabled.")

    # ---- actions --------------------------------------------------------------------------
    def click(self, x: int, y: int, button: str = "left") -> None:
        if button not in ("left", "right"):
            raise ValueError("button must be 'left' or 'right'")
        self._check()
        if self.restrict and (self.region is None or not self.region.contains(x, y)):
            raise OutsideRegionError(f"Refusing to click ({x}, {y}) outside the calibrated region.")
        try:
            # travel in short hops so an emergency stop interrupts mid-movement
            steps = 4 if self.move_duration > 0 else 1
            for _ in range(steps):
                self._check()
                self.backend.move(x, y, self.move_duration / steps)
            self._sleep(self.click_delay)
            self._check()
            self.backend.click(x, y, button)
        except EmergencyStop:
            self._estop.set()
            raise
        self.click_count += 1
        self._sleep(self.click_delay)

    def _sleep(self, t: float) -> None:
        end = time.monotonic() + t
        while time.monotonic() < end:
            self._check()
            time.sleep(min(0.02, max(0.0, end - time.monotonic())))
