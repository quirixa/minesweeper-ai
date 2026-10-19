"""Startup diagnostics: session type, screen capture and mouse-automation availability.

Nothing here clicks or moves the mouse; mouse support is checked by *reading* the pointer position.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np

WAYLAND_HELP = (
    "You are on a Wayland session. GNOME's Wayland compositor deliberately blocks ordinary programs "
    "from reading other windows' pixels and from synthesising mouse input (a security feature that "
    "stops any app from spying on or controlling the desktop). MSS and PyAutoGUI therefore cannot work "
    "reliably here. Supported alternative: log out and choose 'Ubuntu on Xorg' from the gear icon on "
    "the login screen (this affects only that session; no security setting is disabled in Wayland). "
    "A portal-based capture backend is NOT implemented in this project."
)


@dataclass
class Diagnostics:
    session_type: str = "unknown"
    capture_ok: bool = False
    mouse_ok: bool = False
    messages: list[str] = field(default_factory=list)
    xdotool: bool = False

    @property
    def ready(self) -> bool:
        return self.capture_ok and self.mouse_ok

    def report(self) -> str:
        head = f"Session: {self.session_type} | capture: {'OK' if self.capture_ok else 'FAILED'} | " \
               f"mouse control: {'OK' if self.mouse_ok else 'FAILED'}"
        return "\n".join([head] + self.messages)


def detect_session_type() -> str:
    t = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if t in ("x11", "wayland"):
        return t
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def run_diagnostics(capture_backend=None) -> Diagnostics:
    d = Diagnostics(session_type=detect_session_type(), xdotool=shutil.which("xdotool") is not None)
    if d.session_type == "wayland":
        d.messages.append(WAYLAND_HELP)
    if not os.environ.get("DISPLAY"):
        d.messages.append("DISPLAY is not set: no X11/XWayland server reachable, so capture and input "
                          "cannot work from this shell.")
        return d
    # capture check: grab a small area and make sure it is not completely black
    try:
        from .screen_capture import MSSBackend, Region
        backend = capture_backend or MSSBackend()
        desk = backend.desktop()
        img = backend.grab(Region(desk.left, desk.top, min(200, desk.width), min(200, desk.height)))
        d.capture_ok = True
        if np.count_nonzero(img) == 0:
            if d.session_type == "wayland":
                d.messages.append("Capture returned a completely black image: the Wayland compositor is "
                                  "hiding the screen from this program. Use an Xorg session.")
            else:
                d.messages.append("Capture returned an all-black corner. This is normal on an empty virtual "
                                  "display (Xvfb); on a real desktop it would indicate a problem.")
            d.capture_ok = d.session_type != "wayland"
    except Exception as e:  # noqa: BLE001
        d.messages.append(f"Screen capture failed: {e}. If this is X11, check that the 'python3-xlib' "
                          "stack and DISPLAY/XAUTHORITY are available to this process.")
    # mouse check: read-only
    try:
        import pyautogui
        pos = pyautogui.position()
        d.mouse_ok = True
        d.messages.append(f"Mouse position readable at {tuple(pos)} (no movement performed).")
        if d.session_type == "wayland":
            d.messages.append("On Wayland, reading the position succeeds only for XWayland windows; "
                              "clicks into native Wayland windows (e.g. Wayland Firefox) will not work.")
            d.mouse_ok = False
    except Exception as e:  # noqa: BLE001
        d.messages.append(f"Mouse automation unavailable: {e}")
    if not d.xdotool:
        d.messages.append("Optional: install 'xdotool' to let the AI pause when you switch windows.")
    return d


def active_window_title() -> str | None:
    """Focused window title via xdotool (X11 only); None if unavailable."""
    if not shutil.which("xdotool"):
        return None
    try:
        out = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True,
                             text=True, timeout=1.0)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
