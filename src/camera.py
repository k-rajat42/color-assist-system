"""
src/camera.py
=============
Thin wrapper around cv2.VideoCapture.
Handles camera open failures, frame-read failures, and codec/resolution setup.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from config.settings import AppConfig

logger = logging.getLogger(__name__)


class Camera:
    """
    Context-manager-friendly camera abstraction.

    Usage
    -----
    with Camera(config) as cam:
        for frame in cam:
            process(frame)
    """

    def __init__(self, config: AppConfig) -> None:
        self._src    = config.camera_source
        self._width  = config.frame_width
        self._height = config.frame_height
        self._fps    = config.target_fps
        self._cap: cv2.VideoCapture | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── Iterator ──────────────────────────────────────────────────────────────

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    # ── Public API ────────────────────────────────────────────────────────────

    def open(self, retries: int = 3, delay: float = 0.8) -> None:
        """Open the capture device.  Retries `retries` times before raising."""
        for attempt in range(1, retries + 1):
            self._cap = cv2.VideoCapture(self._src)
            if self._cap.isOpened():
                self._configure()
                logger.info(
                    "Camera %d opened  (%dx%d @ %d fps)",
                    self._src, self._width, self._height, self._fps
                )
                return
            logger.warning(
                "Camera %d not available (attempt %d/%d). Retrying…",
                self._src, attempt, retries
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Cannot open camera source {self._src}. "
            "Check that the webcam is connected and not in use by another app."
        )

    def read(self) -> np.ndarray | None:
        """Read one frame; returns None on failure."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("Frame read failed — camera may have disconnected.")
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("Camera released.")

    # ── Private ───────────────────────────────────────────────────────────────

    def _configure(self) -> None:
        assert self._cap is not None
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        # MJPEG gives higher FPS on most USB webcams
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
