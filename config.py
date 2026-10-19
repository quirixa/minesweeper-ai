"""Application settings and on-disk locations."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "minesweeper-ai"


@dataclass
class Settings:
    fps: float = 4.0                 # analysis rate (frames per second)
    click_delay: float = 0.15        # pause before/after each click (seconds)
    move_duration: float = 0.25      # mouse travel time (seconds)
    total_mines: int | None = None   # None = unknown
    min_confidence: float = 0.80     # refuse to act below this recognition confidence
    stable_frames: int = 2           # consecutive identical frames required
    stable_timeout: float = 5.0      # seconds to wait for the board to settle
    manual_mode: bool = True         # ask before each click
    approve_guesses_in_auto: bool = True  # even in auto mode, ask before any guess
    flag_mines: bool = True          # right-click proven mines
    allow_guesses: bool = True
    restrict_to_region: bool = True  # never click outside the calibrated board
    check_active_window: bool = True # pause if the focused window changes (needs xdotool)

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_DIR / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or CONFIG_DIR / "settings.json"
        s = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                names = {f.name for f in fields(cls)}
                for k, v in data.items():
                    if k in names:
                        setattr(s, k, v)
            except (OSError, ValueError):
                pass
        return s
