"""
tests/test_detector.py
======================
Unit tests for ColorDetector (no webcam required — uses synthetic frames).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config.settings import AppConfig
from src.detector import ColorDetector, Detection


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(confidence_threshold=0.0, min_contour_area=100)


@pytest.fixture
def detector(config) -> ColorDetector:
    return ColorDetector(config)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_color_frame(bgr: tuple, size=(480, 640)) -> np.ndarray:
    """Create a solid-color BGR frame."""
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def make_patch_frame(bgr: tuple, size=(480, 640), patch_frac=0.25) -> np.ndarray:
    """Create a frame with a colored rectangle in the centre."""
    frame = np.full((*size, 3), (30, 30, 30), dtype=np.uint8)   # dark background
    h, w = size
    ph, pw = int(h * patch_frac), int(w * patch_frac)
    y0, x0 = (h - ph) // 2, (w - pw) // 2
    frame[y0:y0 + ph, x0:x0 + pw] = bgr
    return frame


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDetectionBasics:
    def test_returns_list(self, detector):
        frame = make_patch_frame((0, 0, 200))   # red-ish in BGR
        result = detector.detect(frame)
        assert isinstance(result, list)

    def test_blue_patch_detected(self, detector):
        """A bright blue patch should yield a Blue detection."""
        frame = make_patch_frame((200, 50, 20))   # BGR blue
        detections = detector.detect(frame)
        colors = [d.color_name for d in detections]
        assert "Blue" in colors, f"Expected Blue; got {colors}"

    def test_green_patch_detected(self, detector):
        frame = make_patch_frame((20, 200, 20))   # BGR green
        detections = detector.detect(frame)
        colors = [d.color_name for d in detections]
        assert "Green" in colors, f"Expected Green; got {colors}"

    def test_no_detection_on_black_frame(self, detector):
        """Pure black frame should not produce noise detections (except Black)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        non_black = [d for d in detections if d.color_name != "Black"]
        assert len(non_black) == 0, f"Unexpected detections: {non_black}"

    def test_detection_fields(self, detector):
        frame = make_patch_frame((0, 200, 0))   # green
        detections = detector.detect(frame)
        for d in detections:
            assert isinstance(d, Detection)
            assert 0.0 <= d.confidence <= 1.0
            x, y, w, h = d.bbox
            assert w > 0 and h > 0
            assert d.area > 0

    def test_sorted_by_area_descending(self, detector):
        frame = make_patch_frame((0, 200, 0))
        detections = detector.detect(frame)
        if len(detections) >= 2:
            areas = [d.area for d in detections]
            assert areas == sorted(areas, reverse=True)


class TestConfidenceScoring:
    def test_confidence_in_range(self, detector):
        frame = make_patch_frame((200, 50, 20))
        for d in detector.detect(frame):
            assert 0.0 <= d.confidence <= 1.0, f"Confidence out of range: {d.confidence}"

    def test_larger_patch_higher_confidence(self, detector):
        """Larger colored region should give higher or equal confidence."""
        small = make_patch_frame((200, 50, 20), patch_frac=0.05)
        large = make_patch_frame((200, 50, 20), patch_frac=0.4)

        small_dets = [d for d in detector.detect(small) if d.color_name == "Blue"]
        large_dets = [d for d in detector.detect(large) if d.color_name == "Blue"]

        if small_dets and large_dets:
            assert large_dets[0].confidence >= small_dets[0].confidence - 0.05


class TestThreshold:
    def test_high_threshold_filters_detections(self):
        cfg = AppConfig(confidence_threshold=0.99, min_contour_area=100)
        det = ColorDetector(cfg)
        frame = make_patch_frame((200, 50, 20), patch_frac=0.05)
        detections = det.detect(frame)
        assert len(detections) == 0, "Expected no detections at 99% threshold"


# ── Standalone runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
