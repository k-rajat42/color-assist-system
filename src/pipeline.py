"""
src/pipeline.py
===============
Orchestrates the full real-time processing loop:
  Camera → Detector → Tracker → VoiceAnnouncer → Renderer → Display

The pipeline is designed to:
  - Be the single place where all modules connect
  - Measure and report FPS accurately
  - Handle graceful shutdown on Q-press or error
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Dict, List, Optional

import cv2

from config.settings import AppConfig
from src.camera    import Camera
from src.detector  import ColorDetector, Detection
from src.renderer  import FrameRenderer
from src.tracker   import ObjectTracker
from src.voice     import VoiceAnnouncer

logger = logging.getLogger(__name__)


class ColorAssistPipeline:
    """End-to-end pipeline for real-time color-assist."""

    # Rolling window for FPS calculation
    _FPS_WINDOW = 30

    def __init__(self, config: AppConfig) -> None:
        self._cfg      = config
        self._detector = ColorDetector(config)
        self._tracker  = ObjectTracker(config)
        self._renderer = FrameRenderer(config)
        self._announcer = VoiceAnnouncer(config)

        self._frame_times: Deque[float] = deque(maxlen=self._FPS_WINDOW)
        self._running = False

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the main loop.  Blocks until the user presses Q."""
        self._running = True
        try:
            with Camera(self._cfg) as cam:
                self._loop(cam)
        finally:
            self._announcer.shutdown()
            cv2.destroyAllWindows()

    # ── Private ───────────────────────────────────────────────────────────────

    def _loop(self, cam: Camera) -> None:
        logger.info("Pipeline running. Press Q in the window to stop.")
        frame_count = 0

        while self._running:
            t0 = time.perf_counter()

            frame = cam.read()
            if frame is None:
                logger.warning("Skipping null frame.")
                continue

            # ── Detection ────────────────────────────────────────────────────
            detections: List[Detection] = self._detector.detect(frame)

            # ── Tracking ─────────────────────────────────────────────────────
            tracked = self._update_trackers(frame, detections)

            # ── Voice ────────────────────────────────────────────────────────
            detected_color_names = [d.color_name for d in tracked]
            self._announcer.announce(detected_color_names)

            # ── Render ───────────────────────────────────────────────────────
            fps = self._measure_fps(t0)
            output = self._renderer.draw_frame(frame, tracked, fps)

            cv2.imshow("Color Assist System", output)

            frame_count += 1
            if frame_count % 90 == 0:
                logger.info(
                    "FPS: %.1f | Detections: %d | Colors: %s",
                    fps,
                    len(tracked),
                    ", ".join(set(detected_color_names)) or "none",
                )

            # ── Quit ─────────────────────────────────────────────────────────
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Q pressed — stopping pipeline.")
                self._running = False

    def _update_trackers(
        self,
        frame,
        detections: List[Detection],
    ) -> List[Detection]:
        """
        For each detection, run the tracker to get a smoothed bbox.
        Returns updated Detection list (bbox may differ slightly from raw detector).
        """
        # Build lookup: color_name → best (highest confidence) detection
        best: Dict[str, Detection] = {}
        for det in detections:
            if det.color_name not in best or det.confidence > best[det.color_name].confidence:
                best[det.color_name] = det

        refined: List[Detection] = []
        for color_name, det in best.items():
            tracked_bbox = self._tracker.update(frame, det, color_name)
            if tracked_bbox is not None:
                # Replace bbox with tracker's smoothed estimate
                x, y, w, h = tracked_bbox
                refined.append(Detection(
                    color_name=det.color_name,
                    confidence=det.confidence,
                    bbox=tracked_bbox,
                    area=det.area,
                    center=(x + w // 2, y + h // 2),
                    mask_fill_ratio=det.mask_fill_ratio,
                ))
        return refined

    def _measure_fps(self, frame_start: float) -> float:
        self._frame_times.append(frame_start)
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            return (len(self._frame_times) - 1) / max(elapsed, 1e-6)
        return 0.0
