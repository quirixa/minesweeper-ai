"""The control loop: observe -> recognise -> solve -> (approve) -> click -> verify. Runs in a worker thread."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from automation.action_verification import Action, verify_action
from automation.mouse_controller import EmergencyStop, MouseController, OutsideRegionError
from capture.screen_capture import CaptureError, ScreenCapture
from capture.session_check import active_window_title
from config import Settings
from solver.board_state import BoardState
from solver.logic_solver import LogicSolver, Move
from vision.calibration import Calibration
from vision.temporal import StabilityTracker, TemporalFilter
from vision.tile_recognition import ProfileMissingError, RecognitionResult, TileRecognizer


class State(Enum):
    IDLE = "Idle"
    RUNNING = "Running"
    PAUSED = "Paused"
    STOPPED = "Stopped"


@dataclass
class Event:
    kind: str            # log | reasoning | state | board | stats | proposal | history
    text: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class Stats:
    revealed: int = 0
    flagged: int = 0
    guesses: int = 0
    wins: int = 0
    losses: int = 0
    clicks: int = 0


class Agent:
    def __init__(self, settings: Settings, capture: ScreenCapture, mouse: MouseController,
                 calibration: Calibration, recognizer: TileRecognizer,
                 emit: Callable[[Event], None] | None = None):
        self.s, self.capture, self.mouse = settings, capture, mouse
        self.calibration, self.recognizer = calibration, recognizer
        self.emit_cb = emit or (lambda e: None)
        self.state = State.IDLE
        self.stats = Stats()
        self.last_result: RecognitionResult | None = None
        self.last_proposal: Move | None = None
        self._stop = threading.Event()
        self._run = threading.Event()
        self._decisions: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._title0: str | None = None
        self._flags_unsupported = False
        self._carry: RecognitionResult | None = None

    # ------------------------------------------------------------------ public controls
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self.mouse.stopped:
            self.log("Emergency stop is active. Reset it before starting.", "warn")
            return
        self._stop.clear()
        self._run.set()
        self._carry = None
        self._flags_unsupported = False
        self.capture.set_region(self.calibration.region)  # geometry is relative to this region
        self.mouse.region = self.calibration.region
        self.mouse.restrict = self.s.restrict_to_region
        self.mouse.move_duration, self.mouse.click_delay = self.s.move_duration, self.s.click_delay
        self._set_state(State.RUNNING)
        self._thread = threading.Thread(target=self._thread_main, name="agent", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        if self.state == State.RUNNING:
            self._run.clear()
            self._set_state(State.PAUSED)
            self.log("Paused by user.")

    def resume(self) -> None:
        if self.state == State.PAUSED and not self.mouse.stopped:
            self._carry = None
            self._run.set()
            self._set_state(State.RUNNING)
            self.log("Resumed.")

    def stop(self) -> None:
        self._stop.set()
        self._run.set()
        self._decisions.put("reject")
        self._set_state(State.STOPPED)

    def emergency_stop(self) -> None:
        self.mouse.emergency_stop()  # first: no further input can be sent, even mid-click
        self.stop()
        self.log("EMERGENCY STOP: automated mouse input is disabled.", "error")

    def reset_emergency_stop(self) -> None:
        self.mouse.reset_emergency_stop()
        self._set_state(State.IDLE)

    def approve(self) -> None:
        self._decisions.put("approve")

    def reject(self) -> None:
        self._decisions.put("reject")

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ------------------------------------------------------------------ events
    def emit(self, kind: str, text: str = "", **data) -> None:
        self.emit_cb(Event(kind, text, data))

    def log(self, text: str, level: str = "info") -> None:
        self.emit("log", text, level=level)

    def _set_state(self, st: State) -> None:
        self.state = st
        self.emit("state", st.value)

    def _pause(self, why: str) -> None:
        self.log(why, "warn")
        self.emit("reasoning", why)
        self._run.clear()
        self._set_state(State.PAUSED)

    def _finish(self, text: str) -> None:
        self.emit("reasoning", text)
        self.log(text)
        self._set_state(State.STOPPED)
        self._stop.set()

    def _gate(self) -> bool:
        """Block while paused; return False if stopped."""
        while not self._run.is_set():
            if self._stop.is_set():
                return False
            time.sleep(0.05)
        return not self._stop.is_set() and not self.mouse.stopped

    # ------------------------------------------------------------------ worker
    def _thread_main(self) -> None:
        try:
            self._loop()
        except EmergencyStop:
            self._set_state(State.STOPPED)
            self.log("Emergency stop: automated input halted.", "error")
        except CaptureError as e:
            self._pause(f"Screen capture problem: {e}")
        except Exception as e:  # noqa: BLE001 - never die silently in a thread
            self._pause(f"Internal error, paused for safety: {type(e).__name__}: {e}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._gate():
                return
            obs = self._carry or self._observe()
            self._carry = None
            if obs is None:
                continue
            board = BoardState(obs.grid)
            self._publish(board, obs)
            if board.is_lost:
                self.stats.losses += 1
                self._publish_stats()
                self._finish("Game lost. Stopping automated input.")
                return
            if board.is_won(self.s.total_mines):
                self.stats.wins += 1
                self._publish_stats()
                self._finish("Game won!")
                return
            analysis = LogicSolver(self.s.total_mines).analyze(board)
            for m in analysis.messages:
                self.emit("reasoning", m)
            if analysis.contradiction:
                self._pause("The board looks contradictory (probably misread). Pausing.")
                continue
            move = self._pick(analysis)
            if move is None:
                self._pause("The visible information is insufficient for a safe decision. Pausing.")
                continue
            self.last_proposal = move
            self.emit("reasoning", move.explanation)
            if not self._authorize(move):
                continue
            self._execute(move, obs)

    def _pick(self, analysis) -> Move | None:
        if analysis.safe_moves:
            return analysis.safe_moves[0]
        if analysis.mine_moves and self.s.flag_mines and not self._flags_unsupported:
            return analysis.mine_moves[0]
        if analysis.guess and self.s.allow_guesses:
            return analysis.guess
        return None

    def _authorize(self, move: Move) -> bool:
        needs = self.s.manual_mode or (move.is_guess and self.s.approve_guesses_in_auto)
        label = "guess" if move.is_guess else move.kind
        self.emit("proposal", f"Proposed {label} at row {move.cell[0]+1}, column {move.cell[1]+1}",
                  cell=move.cell, action=move.kind, certain=move.certain,
                  probability=move.mine_probability, awaiting=needs)
        if not needs:
            return True
        while not self._stop.is_set():
            if not self._run.is_set():
                return False
            try:
                decision = self._decisions.get(timeout=0.1)
            except queue.Empty:
                continue
            if decision == "approve":
                return True
            if self._stop.is_set():
                return False
            self._pause("Move rejected by user; paused.")
            return False
        return False

    def _execute(self, move: Move, obs: RecognitionResult) -> None:
        # never act on stale information: look again right before clicking
        fresh = self._observe()
        if fresh is None:
            return
        if not np.array_equal(fresh.grid, obs.grid):
            self.log("The board changed after analysis; discarding the old plan and re-analyzing.")
            self._carry = fresh
            return
        x, y = self.calibration.cell_to_desktop(*move.cell)
        button = "left" if move.kind == "reveal" else "right"
        self.emit("log", f"Clicking {button} at row {move.cell[0]+1}, col {move.cell[1]+1} -> screen ({x}, {y})",
                  level="info")
        try:
            self.mouse.click(x, y, button)
        except OutsideRegionError as e:
            self._pause(str(e))
            return
        self.stats.clicks += 1
        if self._title0 is None and self.s.check_active_window:
            self._title0 = active_window_title()
        after = self._observe_after(obs.grid)
        if after is None:
            return
        vr = verify_action(obs.grid, after.grid, Action(move.kind, move.cell), self.s.total_mines)
        self.emit("history", "", cell=move.cell, action=move.kind, certain=move.certain,
                  probability=move.mine_probability, outcome=vr.status, rule=move.rule)
        if vr.status in ("ok", "won", "lost"):
            if move.is_guess:
                self.stats.guesses += 1
            self.log(f"The previous click was verified ({vr.detail}) Updating the board model.")
            self._carry = after
        elif vr.status == "no_change":
            if move.kind == "flag":
                self._flags_unsupported = True
                self.log("Right-click did not place a flag on this site; automatic flagging disabled.", "warn")
                self._carry = after
            else:
                self._pause(f"Click had no visible effect. {vr.detail}")
        else:
            self._pause(f"Unexpected board change after click: {vr.detail}")
        self._publish_stats()

    # ------------------------------------------------------------------ observation
    def _observe(self) -> RecognitionResult | None:
        """Wait for a stable frame, recognise it over several frames, return a reliable result."""
        geo = self.calibration.geometry
        tracker = StabilityTracker(max(2, self.s.stable_frames))
        temporal = TemporalFilter(max(2, self.s.stable_frames))
        deadline = time.monotonic() + self.s.stable_timeout
        last: RecognitionResult | None = None
        prev_frame = None
        while time.monotonic() < deadline:
            if self._stop.is_set() or not self._run.is_set():
                return None
            self.capture.fps = self.s.fps
            self.capture.throttle()
            frame = self.capture.grab()
            if self.s.check_active_window and self._title0:
                now = active_window_title()
                if now is not None and now != self._title0:
                    self._pause(f"The focused window changed ('{now}'). Pausing so I don't click elsewhere.")
                    return None
            if not tracker.update(frame):
                temporal.reset()
                continue
            try:
                res = temporal.add(self.recognizer.recognize(frame, geo))
            except ProfileMissingError as e:
                self._pause(str(e))
                return None
            last = res
            self.last_result = res
            if res.is_reliable(self.s.min_confidence):
                return res
        self._publish_low_confidence(last)
        return None

    def _observe_after(self, before: np.ndarray) -> RecognitionResult | None:
        """After a click, wait (briefly) for the board to actually change."""
        end = time.monotonic() + 1.5
        res = None
        while True:
            res = self._observe()
            if res is None or not np.array_equal(res.grid, before) or time.monotonic() > end:
                return res
            time.sleep(0.15)

    # ------------------------------------------------------------------ publishing
    def _publish(self, board: BoardState, res: RecognitionResult) -> None:
        self.stats.revealed, self.stats.flagged = board.revealed_count, board.flag_count
        self.emit("board", "", grid=board.grid.copy(), confidence=res.confidence.copy())
        self._publish_stats()

    def _publish_stats(self) -> None:
        self.emit("stats", "", **self.stats.__dict__)

    def _publish_low_confidence(self, res: RecognitionResult | None) -> None:
        if res is None:
            self._pause("No stable frame within the timeout (the screen keeps changing). Pausing.")
            return
        bad = res.low_cells(self.s.min_confidence)
        self._pause(f"Recognition confidence is too low ({len(bad)} uncertain cell(s), lowest "
                    f"{res.min_confidence:.0%}). Pausing to prevent an unsafe click.")
