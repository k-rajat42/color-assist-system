"""
src/renderer.py
===============
Draws all visual overlays onto the BGR frame:
  - Semi-transparent filled bounding boxes
  - Color label + confidence badge
  - On-screen HUD (mode, FPS, detection count)
  - Color-blind warning banner (when triggered)
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import AppConfig
from src.detector import Detection


class FrameRenderer:
    """Stateless-ish renderer; call draw_frame() once per frame."""

    _FONT       = cv2.FONT_HERSHEY_SIMPLEX
    _HUD_COLOR  = (220, 220, 220)   # BGR light-gray for HUD text
    _WARN_COLOR = (0, 165, 255)     # BGR orange for warnings
    _WHITE      = (255, 255, 255)
    _BLACK      = (0, 0, 0)

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._label_colors = config.label_colors
        self._font_scale   = config.font_scale
        self._thickness    = config.box_thickness
        self._alpha        = config.overlay_alpha
        self._mode         = config.colorblind_mode.capitalize()
        self._warning_text: Optional[str] = None
        self._warning_until: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def draw_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float,
        warning: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compose the full annotated frame.

        Parameters
        ----------
        frame      : original BGR frame (not modified in-place)
        detections : list of color detections for this frame
        fps        : measured frames per second
        warning    : optional color-blind confusion warning string
        """
        canvas = frame.copy()

        for det in detections:
            self._draw_detection(canvas, det)

        self._draw_hud(canvas, fps, len(detections))

        if warning:
            self._warning_text = warning
            self._warning_until = time.time() + 4.0   # show for 4 s

        if self._warning_text and time.time() < self._warning_until:
            self._draw_warning_banner(canvas, self._warning_text)
        else:
            self._warning_text = None

        return canvas

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_detection(self, canvas: np.ndarray, det: Detection) -> None:
        x, y, w, h = det.bbox
        color_bgr = self._label_colors.get(det.color_name, (200, 200, 200))

        # Semi-transparent fill
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, -1)
        cv2.addWeighted(overlay, self._alpha, canvas, 1 - self._alpha, 0, canvas)

        # Border
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color_bgr, self._thickness)

        # Badge background
        label = f"{det.color_name}  {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label, self._FONT, self._font_scale, 1
        )
        pad = 4
        badge_y1 = max(y - th - baseline - pad * 2, 0)
        badge_y2 = max(y, th + baseline + pad * 2)
        cv2.rectangle(
            canvas,
            (x, badge_y1),
            (x + tw + pad * 2, badge_y2),
            color_bgr,
            -1,
        )

        # Label text
        cv2.putText(
            canvas, label,
            (x + pad, badge_y2 - baseline - pad),
            self._FONT, self._font_scale, self._BLACK, 1, cv2.LINE_AA
        )

        # Center crosshair
        cx, cy = det.center
        cv2.drawMarker(canvas, (cx, cy), color_bgr,
                       cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

    def _draw_hud(
        self, canvas: np.ndarray, fps: float, n_detections: int
    ) -> None:
        h, w = canvas.shape[:2]
        lines = [
            f"Color Assist  |  Mode: {self._mode}",
            f"FPS: {fps:5.1f}  |  Detections: {n_detections}",
            "Press  Q  to quit",
        ]
        y0 = 26
        for i, line in enumerate(lines):
            y = y0 + i * 22
            # Drop shadow for readability on any background
            cv2.putText(canvas, line, (11, y + 1),
                        self._FONT, 0.52, self._BLACK, 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y),
                        self._FONT, 0.52, self._HUD_COLOR, 1, cv2.LINE_AA)

    def _draw_warning_banner(self, canvas: np.ndarray, text: str) -> None:
        h, w = canvas.shape[:2]
        banner_h = 36
        # Semi-opaque dark bar at bottom
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        (tw, th), _ = cv2.getTextSize(text, self._FONT, 0.58, 1)
        tx = (w - tw) // 2
        cv2.putText(
            canvas, text,
            (tx, h - 10),
            self._FONT, 0.58, self._WARN_COLOR, 1, cv2.LINE_AA
        )
