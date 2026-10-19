# Minesweeper AI: offline, screen-watching, real mouse

Watches your screen, reads a Minesweeper board **from pixels only**, solves it with local logic and exact
probability counting, explains every decision, and clicks with your real mouse. No cloud, no API keys,
no DOM/source access.

## Honest status (what was and wasn't tested)

| Area | Status |
|---|---|
| Solver, constraints, probabilities (brute-force cross-checked) | Unit tested |
| Grid detection + tile/digit recognition | Tested on synthetic boards (3 zoom levels, noise) and a live Tk board |
| Agent loop (stale-board check, approval, verification, e-stop, low-confidence pause) | Tested end to end against a simulator: 6/6 simulated games won, every "certain" move correct |
| MSS capture + PyAutoGUI clicks | Tested **only on a virtual X11 display (Xvfb)** with the bundled practice board |
| GUI | Builds and handles events under Xvfb; not visually reviewed on a real desktop |
| **Your friend's website, Firefox/Chromium, real GNOME session** | **Not tested.** Expect to calibrate; see Limitations |
| Wayland | Not supported (see below). No portal backend is implemented |

## Install (Ubuntu 22.04 / 24.04, Python >= 3.10)

```bash
sudo apt update
sudo apt install $(grep -v '^#' requirements-system.txt)
cd minesweeper-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --diagnose      # read-only check: nothing is clicked or moved
python main.py                 # launch the GUI
```

If the venv can't import tkinter, `sudo apt install python3-tk` (tkinter comes from apt, not pip).

## X11 vs Wayland (important on GNOME)

GNOME on Wayland deliberately stops ordinary programs from reading other windows or injecting mouse
input, so MSS and PyAutoGUI can't work reliably there. This is a security feature; don't try to defeat it.
Supported route: at the login screen click the gear icon and choose **"Ubuntu on Xorg"** (affects only that
login session; no security setting is disabled). Check with `echo $XDG_SESSION_TYPE` (should say `x11`).
Capture and mouse code are separate classes (`CaptureBackend`, `MouseBackend`) so a portal/other backend
can be added later; none exists yet.

## Calibrate (GUI)

1. Open your Minesweeper page with a **fresh game** (nothing revealed). Keep it visible, not scrolled.
2. **Select region**: drag a rectangle around the board. It runs auto-detection (row/col count and cell size).
   If unreliable, drag *tightly* around the board and use **Manual rows/cols**.
3. **Learn fresh board**: teaches what a covered tile looks like (refused if tiles aren't identical).
4. Play a few moves by hand. If a number is misread, use **Teach digit** (give row, column, value).
   Default colours follow the classic palette; teaching overrides them and also stores the shape.
5. The preview overlays recognised values; green = confident, orange/red = below threshold.

Changing browser zoom or window size means re-doing steps 2-3. Calibration/profile files live in
`~/.config/minesweeper-ai/`.

## Run

- **Manual-approval mode (default)**: every proposed move shows in the reasoning window and the AI waits for
  **Approve move** / **Reject**.
- **Automatic mode**: pick "Automatic" in the right panel. Certain moves execute without asking; guesses still
  ask first unless you untick "Ask before guesses".
- Start always asks for confirmation. **Stop**, **EMERGENCY STOP / Esc**, or slamming the pointer into a
  screen corner (PyAutoGUI failsafe) halts input immediately. Esc works when the AI window has focus; the
  button always works.
- Clicks are restricted to the calibrated region (toggle in settings). It pauses on low confidence, a
  board that changed after analysis, an unexpected change after a click, a click with no effect, capture errors,
  and (if `xdotool` is installed) when the focused window changes.

Try it with no website: `python -m tools.practice_board` (left = reveal, right = flag).

## Solver

1. **Constraints**: each revealed number N with F flagged neighbours becomes "exactly N-F mines among its
   covered neighbours". Impossible numbers raise a contradiction, and the AI refuses to move.
2. **Simple rule**: remaining 0 -> all safe; remaining = covered count -> all mines.
3. **Subset rule**: if A's cells ⊂ B's cells, the difference holds `B - A` mines.
4. **Exhaustive enumeration**: frontier cells are split into independent components; each is enumerated with
   backtracking, counting layouts per mine count and per cell. Components are combined; if the total mine
   count is set, each combination is weighted by C(M, R-s) for M non-frontier cells and R remaining mines.
   Exact integer arithmetic. A cell with 0% or 100% over all layouts is a *proof*.
5. **Guess**: only if nothing is provable; picks the lowest mine probability, labelled `GUESS`, never "safe".
   Without a total-mine count, probabilities of non-frontier cells are an estimate (marked as such), and
   frontier layouts are weighted equally across mine counts.
Explanations are generated from the numbers/constraints that produced each move.

## Tests

```bash
source .venv/bin/activate
python -m pytest -q                                   # ~2 min (end-to-end simulated games are slow)
python -m pytest -q tests/test_solver.py tests/test_probability.py tests/test_constraints.py   # fast
xvfb-run -a -s "-screen 0 1280x800x24" python -m pytest -q tests/test_gui_smoke.py
xvfb-run -a -s "-screen 0 1280x800x24" python -m tools.xvfb_selftest 3   # real capture + real clicks, virtual display
```
No unit test uses a real mouse; mouse tests use fake backends.

## Troubleshooting

- **"DISPLAY is not set" / capture fails**: run from a terminal inside the desktop session, not over SSH.
- **Black screenshots / mouse does nothing**: you're on Wayland; switch to "Ubuntu on Xorg".
- **Clicks land offset**: HiDPI/fractional scaling isn't handled. Set display scale to 100% (or 200% integer) and re-calibrate.
- **Paused: confidence too low**: re-learn the fresh board, Teach digit, enlarge the board (browser zoom), or lower
  "Min confidence" cautiously. Don't set it low to force progress.
- **Right-click doesn't flag**: the AI detects this, disables flagging and continues (not needed to win).
- **No pause on window switch**: install `xdotool`.

## Known limitations

- Digit recognition is colour/shape sampling; themes where all digits look alike need "Teach digit".
- Mine/flag detection is heuristic (dark compact blob = mine). Win/loss detection relies on visible mines or
  no covered cells left; "face" icons and red explosion backgrounds aren't read.
- Question-mark tiles, animated/skinned boards, and very small cells (< ~14 px) are unsupported.
- Active-window pause is best-effort and X11-only.
- Large frontiers (> ~3M search nodes) fall back to simple rules plus a rough estimate, and say so.
- Calibration assumes the board doesn't move/scroll while running.
- Windows/macOS aren't targeted.
