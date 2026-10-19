"""Tkinter desktop application. All agent/capture work happens in worker threads; the GUI thread only
drains an event queue (`after`) and draws, so it never freezes and never touches Tk from other threads."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np

from agent import Agent, Event, State, Stats
from automation.mouse_controller import MouseController, PyAutoGUIBackend
from capture.screen_capture import CaptureError, Region, ScreenCapture
from capture.session_check import run_diagnostics
from config import Settings
from interface.reasoning_panel import ReasoningPanel
from solver.board_state import COVERED, FLAG, MINE
from vision.board_detection import detect_grid
from vision.calibration import (Calibration, auto_calibrate, load_recognizer, manual_calibrate,
                                profile_path)
from vision.tile_recognition import ProfileMissingError

BG, PANEL, FG, MUTED, ACCENT = "#15181e", "#1d2129", "#e6e9ef", "#8b93a1", "#4fc3f7"
PREVIEW_W, PREVIEW_H = 640, 420


def _ppm(img_bgr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes()


class RegionSelector(tk.Toplevel):
    """Shows a snapshot of the desktop; the user drags a rectangle around the game."""

    def __init__(self, master, frame: np.ndarray, desktop: Region, on_done):
        super().__init__(master)
        self.title("Drag a rectangle around the Minesweeper board (Esc cancels)")
        self.on_done, self.desktop = on_done, desktop
        sw, sh = self.winfo_screenwidth() * 0.9, self.winfo_screenheight() * 0.85
        self.scale = min(sw / frame.shape[1], sh / frame.shape[0], 1.0)
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.photo = tk.PhotoImage(data=_ppm(small))
        self.canvas = tk.Canvas(self, width=small.shape[1], height=small.shape[0], cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.rect = None
        self.start = (0, 0)
        self.canvas.bind("<ButtonPress-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._up)
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()

    def _down(self, e):
        self.start = (e.x, e.y)
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#ff5252", width=2)

    def _drag(self, e):
        self.canvas.coords(self.rect, *self.start, e.x, e.y)

    def _up(self, e):
        x0, x1 = sorted((self.start[0], e.x))
        y0, y1 = sorted((self.start[1], e.y))
        if x1 - x0 < 30 or y1 - y0 < 30:
            return
        s = self.scale
        region = Region(self.desktop.left + int(x0 / s), self.desktop.top + int(y0 / s),
                        int((x1 - x0) / s), int((y1 - y0) / s))
        self.destroy()
        self.on_done(region)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Minesweeper AI (offline, screen-watching)")
        self.configure(bg=BG)
        self.settings = Settings.load()
        self.events: queue.Queue = queue.Queue()
        self.capture = ScreenCapture(fps=self.settings.fps)
        self.stats = Stats()
        self.calibration: Calibration | None = None
        self.recognizer = load_recognizer()
        self.agent: Agent | None = None
        self.mouse: MouseController | None = None
        self.mouse_error: str | None = None
        self.last_board = None        # (grid, confidence)
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._closing = False
        self._style()
        self._build()
        self.reasoning = ReasoningPanel(self)
        self.diag = run_diagnostics()
        self._log(self.diag.report())
        self._init_mouse()
        threading.Thread(target=self._preview_worker, daemon=True).start()
        self.after(150, self._poll)
        self.after(200, self._draw_preview)
        self.bind("<Escape>", lambda e: self.emergency())
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._set_buttons(State.IDLE)

    # ------------------------------------------------------------------ construction
    def _style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, bordercolor="#2b303b")
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Muted.TLabel", foreground=MUTED)
        st.configure("Panel.TLabel", background=PANEL, foreground=FG)
        st.configure("TButton", background="#2a3140", foreground=FG, padding=6)
        st.map("TButton", background=[("active", "#37425a"), ("disabled", "#20242d")],
               foreground=[("disabled", "#5c6370")])
        st.configure("Go.TButton", background="#2e7d32")
        st.configure("Emerg.TButton", background="#c62828", foreground="white", font=("TkDefaultFont", 11, "bold"))
        st.map("Emerg.TButton", background=[("active", "#e53935")])
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.configure("TRadiobutton", background=BG, foreground=FG)
        st.configure("TSpinbox", fieldbackground=PANEL, foreground=FG)

    def _build(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=8)
        self.btn_start = ttk.Button(bar, text="Start", style="Go.TButton", command=self.on_start)
        self.btn_pause = ttk.Button(bar, text="Pause", command=lambda: self.agent and self.agent.pause())
        self.btn_resume = ttk.Button(bar, text="Resume", command=lambda: self.agent and self.agent.resume())
        self.btn_stop = ttk.Button(bar, text="Stop", command=self.on_stop)
        for b in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop):
            b.pack(side="left", padx=3)
        self.btn_emerg = ttk.Button(bar, text="EMERGENCY STOP (Esc)", style="Emerg.TButton", command=self.emergency)
        self.btn_emerg.pack(side="right", padx=3)
        self.estop_label = tk.Label(bar, text="  Input: ARMED-OFF (idle)  ", bg="#2a3140", fg=FG)
        self.estop_label.pack(side="right", padx=8)
        self.btn_approve = ttk.Button(bar, text="Approve move", command=lambda: self.agent and self.agent.approve())
        self.btn_reject = ttk.Button(bar, text="Reject", command=lambda: self.agent and self.agent.reject())
        self.btn_reject.pack(side="right", padx=3)
        self.btn_approve.pack(side="right", padx=3)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, width=PREVIEW_W, height=PREVIEW_H, bg="#0b0d11", highlightthickness=1,
                                highlightbackground="#2b303b")
        self.canvas.pack()
        self._photo = None
        cal = ttk.Frame(left)
        cal.pack(fill="x", pady=6)
        for text, cmd in (("Select region", self.on_select_region), ("Auto-detect board", self.on_autodetect),
                          ("Manual rows/cols", self.on_manual), ("Learn fresh board", self.on_learn),
                          ("Teach digit", self.on_teach), ("Diagnostics", self.on_diag)):
            ttk.Button(cal, text=text, command=cmd).pack(side="left", padx=2)
        self.info = ttk.Label(left, text="No calibration yet.", style="Muted.TLabel")
        self.info.pack(anchor="w")
        self.logbox = tk.Text(left, height=9, bg="#0f1115", fg=FG, relief="flat", wrap="word", state="disabled")
        self.logbox.pack(fill="both", expand=True, pady=6)
        self.logbox.tag_configure("warn", foreground="#ffb74d")
        self.logbox.tag_configure("error", foreground="#ef5350")

        right = ttk.Frame(body, style="Panel.TFrame")
        right.pack(side="right", fill="y", padx=(10, 0))
        self.stat_vars = {k: tk.StringVar(value="-") for k in
                          ("State", "Board", "Revealed", "Flagged", "Guesses", "Won / Lost", "Confidence", "Current move")}
        for i, (k, v) in enumerate(self.stat_vars.items()):
            ttk.Label(right, text=k, style="Panel.TLabel").grid(row=i, column=0, sticky="w", padx=8, pady=2)
            ttk.Label(right, textvariable=v, style="Panel.TLabel", wraplength=190).grid(row=i, column=1, sticky="w", padx=8)
        r0 = len(self.stat_vars)
        ttk.Separator(right).grid(row=r0, column=0, columnspan=2, sticky="ew", pady=6)
        s = self.settings
        self.v = {"fps": tk.DoubleVar(value=s.fps), "click_delay": tk.DoubleVar(value=s.click_delay),
                  "move_duration": tk.DoubleVar(value=s.move_duration),
                  "min_confidence": tk.DoubleVar(value=s.min_confidence),
                  "total_mines": tk.StringVar(value="" if s.total_mines is None else str(s.total_mines)),
                  "manual": tk.BooleanVar(value=s.manual_mode), "flag": tk.BooleanVar(value=s.flag_mines),
                  "restrict": tk.BooleanVar(value=s.restrict_to_region),
                  "guess_ok": tk.BooleanVar(value=s.approve_guesses_in_auto),
                  "allow_guess": tk.BooleanVar(value=s.allow_guesses)}
        rows = [("Analysis rate (fps)", "fps", 0.5, 20, 0.5), ("Click delay (s)", "click_delay", 0, 3, 0.05),
                ("Mouse move (s)", "move_duration", 0, 3, 0.05), ("Min confidence", "min_confidence", 0.5, 1, 0.05)]
        for i, (lab, key, lo, hi, inc) in enumerate(rows, start=r0 + 1):
            ttk.Label(right, text=lab, style="Panel.TLabel").grid(row=i, column=0, sticky="w", padx=8, pady=2)
            ttk.Spinbox(right, from_=lo, to=hi, increment=inc, textvariable=self.v[key], width=7,
                        command=self.apply_settings).grid(row=i, column=1, sticky="w", padx=8)
        i = r0 + 1 + len(rows)
        ttk.Label(right, text="Total mines (blank=?)", style="Panel.TLabel").grid(row=i, column=0, sticky="w", padx=8)
        ttk.Entry(right, textvariable=self.v["total_mines"], width=8).grid(row=i, column=1, sticky="w", padx=8)
        ttk.Radiobutton(right, text="Manual approval of every move", variable=self.v["manual"], value=True,
                        command=self.apply_settings).grid(row=i + 1, column=0, columnspan=2, sticky="w", padx=8)
        ttk.Radiobutton(right, text="Automatic (validated moves)", variable=self.v["manual"], value=False,
                        command=self.apply_settings).grid(row=i + 2, column=0, columnspan=2, sticky="w", padx=8)
        for j, (txt, key) in enumerate((("Flag proven mines", "flag"), ("Restrict clicks to board", "restrict"),
                                        ("Ask before guesses (auto mode)", "guess_ok"), ("Allow guesses", "allow_guess"))):
            ttk.Checkbutton(right, text=txt, variable=self.v[key], command=self.apply_settings).grid(
                row=i + 3 + j, column=0, columnspan=2, sticky="w", padx=8)
        ttk.Button(right, text="Show reasoning window", command=lambda: self.reasoning.deiconify()).grid(
            row=i + 8, column=0, columnspan=2, pady=10, padx=8, sticky="ew")

    # ------------------------------------------------------------------ helpers
    def _log(self, text: str, level: str = "info"):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", time.strftime("%H:%M:%S ") + text + "\n", level)
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def apply_settings(self):
        s = self.settings
        try:
            s.fps = float(self.v["fps"].get())
            s.click_delay = float(self.v["click_delay"].get())
            s.move_duration = float(self.v["move_duration"].get())
            s.min_confidence = float(self.v["min_confidence"].get())
            tm = self.v["total_mines"].get().strip()
            s.total_mines = int(tm) if tm else None
        except (ValueError, tk.TclError):
            self._log("Some settings are not valid numbers; keeping previous values.", "warn")
        s.manual_mode = bool(self.v["manual"].get())
        s.flag_mines = bool(self.v["flag"].get())
        s.restrict_to_region = bool(self.v["restrict"].get())
        s.approve_guesses_in_auto = bool(self.v["guess_ok"].get())
        s.allow_guesses = bool(self.v["allow_guess"].get())
        self.capture.fps = s.fps
        if self.mouse:
            self.mouse.restrict = s.restrict_to_region
            self.mouse.move_duration, self.mouse.click_delay = s.move_duration, s.click_delay
        s.save()

    def _init_mouse(self):
        try:
            self.mouse = MouseController(PyAutoGUIBackend(), None, self.settings.restrict_to_region,
                                         self.settings.move_duration, self.settings.click_delay)
        except Exception as e:  # noqa: BLE001
            self.mouse_error = str(e)
            self._log(f"Mouse backend unavailable: {e}", "error")

    def _set_buttons(self, st: State):
        running, paused = st == State.RUNNING, st == State.PAUSED
        can_start = st in (State.IDLE, State.STOPPED) and self.mouse is not None
        self.btn_start.state(["!disabled"] if can_start else ["disabled"])
        self.btn_pause.state(["!disabled"] if running else ["disabled"])
        self.btn_resume.state(["!disabled"] if paused else ["disabled"])
        self.btn_stop.state(["!disabled"] if (running or paused) else ["disabled"])
        halted = bool(self.mouse and self.mouse.stopped)
        self.estop_label.configure(
            text="  INPUT DISABLED (emergency stop)  " if halted else
            ("  Input: ACTIVE  " if running else "  Input: idle  "),
            bg="#c62828" if halted else ("#2e7d32" if running else "#2a3140"))

    # ------------------------------------------------------------------ preview
    def _preview_worker(self):
        while not self._closing:
            try:
                self.capture.set_region(self.calibration.region if self.calibration else None)
                frame = self.capture.grab(retries=0)
                with self._frame_lock:
                    self._latest_frame = frame
            except CaptureError as e:
                with self._frame_lock:
                    self._latest_frame = None
                self.events.put(Event("log", f"Preview: {e}", {"level": "warn"}))
                time.sleep(2)
            time.sleep(0.25)

    def _draw_preview(self):
        if self._closing:
            return
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is not None:
            scale = min(PREVIEW_W / frame.shape[1], PREVIEW_H / frame.shape[0])
            if self.calibration and self.last_board is not None:
                self._overlay(frame)
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self._photo = tk.PhotoImage(data=_ppm(small))
            self.canvas.delete("all")
            self.canvas.create_image(PREVIEW_W // 2, PREVIEW_H // 2, image=self._photo)
        self.after(250, self._draw_preview)

    def _overlay(self, frame):
        geo = self.calibration.geometry
        grid, conf = self.last_board
        if grid.shape != (geo.rows, geo.cols):
            return
        glyph = {COVERED: "", FLAG: "F", MINE: "*"}
        for r in range(geo.rows):
            for c in range(geo.cols):
                x0, y0, x1, y1 = geo.cell_rect(r, c)
                cf = conf[r, c]
                col = (80, 200, 80) if cf >= self.settings.min_confidence else (0, 140, 255) if cf > 0.5 else (60, 60, 230)
                cv2.rectangle(frame, (x0, y0), (x1, y1), col, 1)
                t = glyph.get(int(grid[r, c]), str(int(grid[r, c]) or "."))
                if t:
                    cv2.putText(frame, t, (x0 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # ------------------------------------------------------------------ calibration actions
    def _fresh_frame(self):
        try:
            return self.capture.backend.grab(self.calibration.region if self.calibration else None)
        except CaptureError as e:
            messagebox.showerror("Capture failed", str(e))
            return None

    def on_select_region(self):
        try:
            desk = self.capture.backend.desktop()
            frame = self.capture.backend.grab(desk)
        except CaptureError as e:
            messagebox.showerror("Capture failed", f"{e}\n\n{self.diag.report()}")
            return
        RegionSelector(self, frame, desk, self._region_chosen)

    def _region_chosen(self, region: Region):
        self.capture.set_region(region)
        self.calibration = None
        self.info.configure(text=f"Region {region.width}x{region.height} at ({region.left},{region.top}). "
                                 "Now press 'Auto-detect board' (or 'Manual rows/cols').")
        self._pending_region = region
        self.on_autodetect()

    def on_autodetect(self):
        region = getattr(self, "_pending_region", None) or (self.calibration.region if self.calibration else None)
        if region is None:
            messagebox.showinfo("Select region first", "Use 'Select region' to drag around the board.")
            return
        frame = self.capture.backend.grab(region)
        cal = auto_calibrate(frame, region)
        if cal is None or cal.geometry.confidence < 0.4:
            self.info.configure(text="Auto-detection unreliable; use 'Manual rows/cols'.")
            self._log("Auto-detection failed or low confidence. Select a tighter region and use Manual rows/cols.", "warn")
            return
        self._set_calibration(cal, "auto")

    def on_manual(self):
        region = getattr(self, "_pending_region", None) or (self.calibration.region if self.calibration else None)
        if region is None:
            messagebox.showinfo("Select region first", "Drag a rectangle tightly around the board first.")
            return
        rows = simpledialog.askinteger("Rows", "Number of rows:", parent=self, minvalue=1, maxvalue=100)
        cols = simpledialog.askinteger("Columns", "Number of columns:", parent=self, minvalue=1, maxvalue=100) if rows else None
        if not rows or not cols:
            return
        self._set_calibration(manual_calibrate(region, 0, 0, region.width, region.height, rows, cols), "manual")

    def _set_calibration(self, cal: Calibration, how: str):
        self.calibration = cal
        g = cal.geometry
        self.info.configure(text=f"Board {g.rows} rows x {g.cols} cols ({how}, cell {g.cell_w:.1f}px, "
                                 f"grid confidence {g.confidence:.0%}). Start a NEW game and press 'Learn fresh board'.")
        self.stat_vars["Board"].set(f"{g.rows} x {g.cols}")
        cal.save()
        self._log(f"Calibration saved ({how}): {g.rows}x{g.cols}.")

    def on_learn(self):
        if not self.calibration:
            messagebox.showinfo("Calibrate first", "Select the region and detect the board first.")
            return
        frame = self._fresh_frame()
        if frame is None:
            return
        try:
            n = self.recognizer.learn_covered(frame, self.calibration.geometry)
        except ProfileMissingError as e:
            messagebox.showwarning("Could not learn tiles", str(e))
            return
        self.recognizer.profile.save(profile_path())
        self._log(f"Learned the covered-tile look from {n} cells.")

    def on_teach(self):
        if not self.calibration:
            return
        frame = self._fresh_frame()
        if frame is None:
            return
        r = simpledialog.askinteger("Teach digit", "Row of a revealed numbered cell (1-based):", parent=self)
        c = simpledialog.askinteger("Teach digit", "Column (1-based):", parent=self) if r else None
        d = simpledialog.askinteger("Teach digit", "The number shown in that cell (1-8):", parent=self,
                                    minvalue=1, maxvalue=8) if c else None
        if not (r and c and d):
            return
        try:
            self.recognizer.teach_digit(frame, self.calibration.geometry, (r - 1, c - 1), d)
        except Exception as e:  # noqa: BLE001
            messagebox.showwarning("Teach digit", str(e))
            return
        self.recognizer.profile.save(profile_path())
        self._log(f"Learned how a {d} looks.")

    def on_diag(self):
        self.diag = run_diagnostics()
        messagebox.showinfo("Diagnostics (no clicks performed)", self.diag.report())

    # ------------------------------------------------------------------ run control
    def on_start(self):
        if not self.calibration or self.recognizer.profile.covered_sig is None:
            messagebox.showwarning("Not calibrated", "Select the region, detect the board and press "
                                   "'Learn fresh board' (on a new game) first.")
            return
        if not self.diag.mouse_ok or not self.diag.capture_ok:
            if not messagebox.askyesno("Diagnostics failed", self.diag.report() + "\n\nStart anyway?"):
                return
        self.apply_settings()
        if not messagebox.askokcancel(
                "Confirm mouse control",
                "The AI will move and click your REAL mouse inside the calibrated board region.\n\n"
                "Stop anytime with Stop, the EMERGENCY STOP button / Esc, or by slamming the pointer into a "
                "screen corner (PyAutoGUI failsafe).\n\nStart?"):
            return
        self.mouse.reset_emergency_stop()
        self.agent = Agent(self.settings, self.capture, self.mouse, self.calibration, self.recognizer,
                           self.events.put)
        self.agent.stats = self.stats
        self.agent.start()

    def on_stop(self):
        if self.agent:
            self.agent.stop()

    def emergency(self):
        if self.agent:
            self.agent.emergency_stop()
        elif self.mouse:
            self.mouse.emergency_stop()
        self._set_buttons(State.STOPPED)

    # ------------------------------------------------------------------ event pump
    def _poll(self):
        try:
            for _ in range(200):
                ev: Event = self.events.get_nowait()
                self._handle(ev)
        except queue.Empty:
            pass
        if not self._closing:
            self.after(100, self._poll)

    def _handle(self, ev: Event):
        d = ev.data
        if ev.kind == "log":
            self._log(ev.text, d.get("level", "info"))
        elif ev.kind == "reasoning":
            self.reasoning.add_reasoning(ev.text)
        elif ev.kind == "state":
            self.stat_vars["State"].set(ev.text)
            self._set_buttons(State(ev.text))
        elif ev.kind == "board":
            self.last_board = (d["grid"], d["confidence"])
            g, c = d["grid"], d["confidence"]
            self.stat_vars["Board"].set(f"{g.shape[0]} x {g.shape[1]}")
            self.stat_vars["Confidence"].set(f"min {c.min():.0%}, mean {c.mean():.0%}")
        elif ev.kind == "stats":
            self.stat_vars["Revealed"].set(str(d["revealed"]))
            self.stat_vars["Flagged"].set(str(d["flagged"]))
            self.stat_vars["Guesses"].set(str(d["guesses"]))
            self.stat_vars["Won / Lost"].set(f"{d['wins']} / {d['losses']}")
        elif ev.kind == "proposal":
            self.stat_vars["Current move"].set(ev.text)
            if d.get("awaiting"):
                self.btn_approve.state(["!disabled"])
                self.btn_reject.state(["!disabled"])
                self._log(ev.text + " - waiting for your approval.")
            else:
                self.btn_approve.state(["disabled"])
                self.btn_reject.state(["disabled"])
        elif ev.kind == "history":
            self.reasoning.add_history(d)

    def _close(self):
        self._closing = True
        if self.agent:
            self.agent.stop()
        self.destroy()
