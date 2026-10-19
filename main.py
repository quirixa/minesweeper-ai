"""Entry point.  `python main.py` opens the GUI; `python main.py --diagnose` only checks capture/mouse
availability (read-only: nothing is clicked or moved)."""
from __future__ import annotations

import argparse
import sys


def diagnose(snapshot: str | None) -> int:
    from capture.session_check import run_diagnostics
    d = run_diagnostics()
    print(d.report())
    if snapshot and d.capture_ok:
        import cv2
        from capture.screen_capture import MSSBackend
        cv2.imwrite(snapshot, MSSBackend().grab())
        print(f"Snapshot written to {snapshot} (only because you asked for it).")
    return 0 if d.ready else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline screen-watching Minesweeper AI")
    ap.add_argument("--diagnose", action="store_true", help="check session/capture/mouse support and exit")
    ap.add_argument("--snapshot", metavar="FILE", help="with --diagnose: save one screenshot to FILE")
    args = ap.parse_args()
    if args.diagnose:
        return diagnose(args.snapshot)
    try:
        from interface.app import App
    except ImportError as e:  # tkinter missing is the common cause
        print(f"Cannot start the GUI: {e}\nOn Ubuntu: sudo apt install python3-tk", file=sys.stderr)
        return 2
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
