"""Calibration: where the board is on screen, its geometry, and which tile profile to use."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capture.screen_capture import Region
from config import CONFIG_DIR
from .board_detection import GridGeometry, detect_grid, geometry_from_manual
from .tile_recognition import TileProfile, TileRecognizer


@dataclass
class Calibration:
    region: Region            # desktop rectangle that is captured
    geometry: GridGeometry    # grid inside that region (pixel coords relative to the region)
    name: str = "default"

    def cell_to_desktop(self, r: int, c: int) -> tuple[int, int]:
        x, y = self.geometry.cell_center(r, c)
        return int(round(self.region.left + x)), int(round(self.region.top + y))

    def save(self, directory: Path | None = None) -> Path:
        d = directory or CONFIG_DIR / "profiles"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.name}.calibration.json"
        path.write_text(json.dumps({"region": self.region.to_dict(), "geometry": self.geometry.to_dict(),
                                    "name": self.name}, indent=1))
        return path

    @classmethod
    def load(cls, name: str = "default", directory: Path | None = None) -> "Calibration":
        path = (directory or CONFIG_DIR / "profiles") / f"{name}.calibration.json"
        d = json.loads(path.read_text())
        return cls(Region.from_dict(d["region"]), GridGeometry.from_dict(d["geometry"]), d.get("name", name))


def auto_calibrate(frame: np.ndarray, region: Region, name: str = "default") -> Calibration | None:
    geo = detect_grid(frame)
    return Calibration(region, geo, name) if geo else None


def manual_calibrate(region: Region, x0, y0, x1, y1, rows, cols, name: str = "default") -> Calibration:
    """x0..y1 are pixel coordinates inside the captured region."""
    return Calibration(region, geometry_from_manual(x0, y0, x1, y1, rows, cols), name)


def profile_path(name: str = "default") -> Path:
    return CONFIG_DIR / "profiles" / f"{name}.tiles.json"


def load_recognizer(name: str = "default") -> TileRecognizer:
    p = profile_path(name)
    return TileRecognizer(TileProfile.load(p) if p.exists() else TileProfile(name=name))
