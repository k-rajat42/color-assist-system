"""
src/tracker.py
==============
Wraps an OpenCV single-object tracker (CSRT / KCF / MOSSE) and adds:
  - Automatic re-initialisation from fresh detections
  - Graceful failure recovery
  - Tracked bbox → Detection bridge
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.settings import AppConfig
from src.detector import Detection

logger = logging.getLogger(__name__)

# Map config string → OpenCV tracker constructor
_TRACKER_FACTORIES: Dict[str, callable] = {
    "CSRT":  cv2.TrackerCSRT_create,
    "KCF":   cv2.TrackerKCF_create,
    "MOSSE": cv2.legacy.TrackerMOSSE_create,
}


class ObjectTracker:
    """
    Maintains one tracker per color.  On each frame:
      1. If a fresh Detection is available → re-init the tracker.
      2. Otherwise → update existing tracker.
      3. Return a (possibly smoothed) bounding box.

    The tracker supplements the detector: it provides smooth bbox interpolation
    between detector frames and handles brief occlusions.
    """

    def __init__(self, config: AppConfig) -> None:
        factory = _TRACKER_FACTORIES.get(config.tracker_type, cv2.TrackerCSRT_create)
        self._factory = factory
        self._reinit_interval = config.tracker_reinit_interval
        self._trackers: Dict[str, cv2.Tracker] = {}
        self._frames_since_reinit: Dict[str, int] = {}
        self._last_bbox: Dict[str, Tuple[int, int, int, int]] = {}
        logger.debug("ObjectTracker initialised (type=%s)", config.tracker_type)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        detection: Optional[Detection],
        color_name: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Update tracker for `color_name`.

        Parameters
        ----------
        frame      : current BGR frame
        detection  : latest Detection from ColorDetector (may be None)
        color_name : which color's tracker to update

        Returns
        -------
        (x, y, w, h) of the tracked bbox, or None if tracking failed.
        """
        should_reinit = (
            detection is not None
            and (
                color_name not in self._trackers
                or self._frames_since_reinit.get(color_name, 999) >= self._reinit_interval
            )
        )

        if should_reinit and detection is not None:
            self._reinit(frame, color_name, detection.bbox)

        if color_name not in self._trackers:
            return detection.bbox if detection else None

        ok, bbox_f = self._trackers[color_name].update(frame)
        self._frames_since_reinit[color_name] = (
            self._frames_since_reinit.get(color_name, 0) + 1
        )

        if ok:
            bbox = tuple(int(v) for v in bbox_f)  # type: ignore[assignment]
            self._last_bbox[color_name] = bbox
            return bbox
        else:
            logger.debug("Tracker lost for '%s'; waiting for re-detection.", color_name)
            self._trackers.pop(color_name, None)
            return None

    def reset(self) -> None:
        """Clear all trackers (e.g., on scene change)."""
        self._trackers.clear()
        self._frames_since_reinit.clear()
        self._last_bbox.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    def _reinit(
        self,
        frame: np.ndarray,
        color_name: str,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        tracker = self._factory()
        tracker.init(frame, bbox)
        self._trackers[color_name] = tracker
        self._frames_since_reinit[color_name] = 0
        logger.debug("Tracker re-initialised for '%s' @ %s", color_name, bbox)
