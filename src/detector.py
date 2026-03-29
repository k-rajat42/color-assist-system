"""
src/detector.py
===============
HSV-based color detector with:
  - Adaptive histogram equalisation (CLAHE) for lighting robustness
  - Morphological noise removal
  - Per-detection confidence scoring
  - Contour-level bounding box extraction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detected color region."""
    color_name: str
    confidence: float                        # 0.0 – 1.0
    bbox: Tuple[int, int, int, int]          # x, y, w, h  (pixel coords)
    area: int                                # contour area in px²
    center: Tuple[int, int]                  # (cx, cy)
    mask_fill_ratio: float                   # filled pixels / bbox area

    def __repr__(self) -> str:
        x, y, w, h = self.bbox
        return (
            f"Detection({self.color_name}, conf={self.confidence:.2f}, "
            f"box=({x},{y},{w},{h}), area={self.area})"
        )


class ColorDetector:
    """
    Detects dominant color regions in an image using per-color HSV masks.

    Pipeline per frame
    ------------------
    1. Resize to processing resolution (for speed)
    2. Gaussian blur  →  reduce sensor noise
    3. Convert BGR → HSV
    4. CLAHE on V-channel  →  lighting normalisation
    5. For each configured color: create mask → morphological open/close
    6. Find external contours; filter by min area
    7. Score each contour to produce a Detection
    """

    # Processing resolution (lower than display for speed)
    PROC_W, PROC_H = 640, 360

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, config.kernel_size
        )
        self._clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_tile_grid,
        )
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        logger.debug("ColorDetector initialised (kernel=%s)", config.kernel_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run the detection pipeline on one BGR frame.

        Returns
        -------
        List[Detection] sorted by area descending (largest object first).
        """
        proc = self._preprocess(frame_bgr)
        hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
        hsv = self._apply_clahe(hsv)

        detections: List[Detection] = []
        for color_name, ranges in self._cfg.color_ranges.items():
            mask = self._build_mask(hsv, ranges)
            mask = self._clean_mask(mask)
            detections.extend(
                self._extract_detections(mask, color_name, frame_bgr.shape)
            )

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    # ── Private helpers ───────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize + blur for consistent processing speed."""
        h, w = frame.shape[:2]
        self._scale_x = w / self.PROC_W
        self._scale_y = h / self.PROC_H
        small = cv2.resize(frame, (self.PROC_W, self.PROC_H),
                           interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, self._cfg.gaussian_blur_kernel, 0)

    def _apply_clahe(self, hsv: np.ndarray) -> np.ndarray:
        """Equalise the Value channel so colors stay detectable in dim/harsh light."""
        h, s, v = cv2.split(hsv)
        v_eq = self._clahe.apply(v)
        return cv2.merge([h, s, v_eq])

    def _build_mask(
        self,
        hsv: np.ndarray,
        ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
    ) -> np.ndarray:
        """Union of all HSV range masks for one color (handles hue wrap-around)."""
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lo = np.array(lower, dtype=np.uint8)
            hi = np.array(upper, dtype=np.uint8)
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
        return combined

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Morphological open (remove noise) then close (fill gaps)."""
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        return closed

    def _extract_detections(
        self,
        mask: np.ndarray,
        color_name: str,
        original_shape: Tuple[int, ...],
    ) -> List[Detection]:
        """Convert contours → Detection objects, scaled back to original resolution."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results: List[Detection] = []
        min_area_proc = self._cfg.min_contour_area / (self._scale_x * self._scale_y)

        for cnt in contours:
            area_proc = cv2.contourArea(cnt)
            if area_proc < min_area_proc:
                continue

            x_p, y_p, w_p, h_p = cv2.boundingRect(cnt)

            # Scale bounding box back to original frame coordinates
            x = int(x_p * self._scale_x)
            y = int(y_p * self._scale_y)
            w = int(w_p * self._scale_x)
            h = int(h_p * self._scale_y)

            area_orig = int(area_proc * self._scale_x * self._scale_y)

            # Confidence: ratio of mask pixels to bbox area (compactness proxy)
            bbox_area_proc = max(w_p * h_p, 1)
            fill_ratio = float(area_proc) / bbox_area_proc
            confidence = self._score_confidence(fill_ratio, area_proc)

            if confidence < self._cfg.confidence_threshold:
                continue

            cx, cy = x + w // 2, y + h // 2
            results.append(Detection(
                color_name=color_name,
                confidence=confidence,
                bbox=(x, y, w, h),
                area=area_orig,
                center=(cx, cy),
                mask_fill_ratio=fill_ratio,
            ))
        return results

    @staticmethod
    def _score_confidence(fill_ratio: float, area_proc: float) -> float:
        """
        Heuristic confidence combining:
          - fill_ratio : how solid/compact the blob is (higher → less noise)
          - size_score : bonus for larger objects (more stable signal)
        Returns a value in [0, 1].
        """
        # Sigmoid-style size normalisation: saturates around 10k px²
        size_score = 1.0 - np.exp(-area_proc / 5_000.0)
        # Weighted blend
        confidence = 0.55 * fill_ratio + 0.45 * size_score
        return float(np.clip(confidence, 0.0, 1.0))
