"""Screen capture behind a small backend interface (so Wayland/portal backends can be added)."""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def to_dict(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        return cls(int(d["left"]), int(d["top"]), int(d["width"]), int(d["height"]))


class CaptureBackend(ABC):
    @abstractmethod
    def grab(self, region: Region | None = None) -> np.ndarray:
        """Return a BGR uint8 image of `region` (desktop pixels) or of the whole virtual desktop."""

    @abstractmethod
    def desktop(self) -> Region:
        """Bounding box of all monitors."""

    def close(self) -> None:  # pragma: no cover
        pass


class MSSBackend(CaptureBackend):
    """X11 capture through MSS. Handles are per-thread because X connections are not thread-safe."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _sct(self):
        if not hasattr(self._local, "sct"):
            try:
                import mss
                self._local.sct = (mss.MSS if hasattr(mss, "MSS") else mss.mss)()
            except Exception as e:  # noqa: BLE001 - surface any backend failure uniformly
                raise CaptureError(f"Cannot open screen capture: {e}") from e
        return self._local.sct

    def desktop(self) -> Region:
        m = self._sct().monitors[0]
        return Region(m["left"], m["top"], m["width"], m["height"])

    def grab(self, region: Region | None = None) -> np.ndarray:
        region = region or self.desktop()
        try:
            shot = self._sct().grab(region.to_dict())
        except Exception as e:  # noqa: BLE001
            # drop the handle so the next call reconnects (e.g. after a display change)
            self._local.__dict__.pop("sct", None)
            raise CaptureError(f"Screen grab failed: {e}") from e
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])  # BGRA -> BGR


class ScreenCapture:
    """Wraps a backend with region handling, FPS limiting and resolution-change detection."""

    def __init__(self, backend: CaptureBackend | None = None, fps: float = 4.0):
        self.backend = backend or MSSBackend()
        self.region: Region | None = None
        self.fps = fps
        self._last = 0.0
        self._desktop_size: tuple[int, int] | None = None

    def set_region(self, region: Region | None) -> None:
        self.region = region

    def desktop_changed(self) -> bool:
        """True if the desktop size differs from the last time this was asked."""
        d = self.backend.desktop()
        size = (d.width, d.height)
        changed = self._desktop_size is not None and size != self._desktop_size
        self._desktop_size = size
        return changed

    def throttle(self) -> None:
        wait = (1.0 / max(self.fps, 0.1)) - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def grab(self, retries: int = 2) -> np.ndarray:
        err: Exception | None = None
        for _ in range(retries + 1):
            try:
                if self.region is not None and self.desktop_changed():
                    raise CaptureError("Screen resolution changed; please re-select the game region.")
                return self.backend.grab(self.region)
            except CaptureError as e:
                err = e
                if "resolution" in str(e):
                    break
                time.sleep(0.1)
        raise CaptureError(str(err))
