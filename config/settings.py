"""
config/settings.py
==================
Central configuration for the Color Assist System.
All tuneable parameters live here — no magic numbers in the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List


# ── HSV color range type alias ─────────────────────────────────────────────────
# Each entry: (lower_hsv, upper_hsv) as numpy-compatible tuples
HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


@dataclass
class AppConfig:
    # ── Camera ────────────────────────────────────────────────────────────────
    camera_source: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30

    # ── Color blindness simulation ────────────────────────────────────────────
    colorblind_mode: str = "deuteranopia"   # deuteranopia | protanopia | tritanopia | normal

    # ── Voice ─────────────────────────────────────────────────────────────────
    voice_enabled: bool = True
    announce_interval: float = 2.5          # seconds between announcements
    tts_rate: int = 165                     # words per minute
    tts_volume: float = 0.9

    # ── Detection ─────────────────────────────────────────────────────────────
    confidence_threshold: float = 0.65
    min_contour_area: int = 2_500           # px²  – filters noise
    kernel_size: Tuple[int, int] = (7, 7)   # morphological kernel
    gaussian_blur_kernel: Tuple[int, int] = (5, 5)

    # ── Adaptive lighting ─────────────────────────────────────────────────────
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker_type: str = "CSRT"              # CSRT | KCF | MOSSE
    tracker_reinit_interval: int = 45       # frames between re-init

    # ── UI overlay ────────────────────────────────────────────────────────────
    box_thickness: int = 2
    font_scale: float = 0.6
    overlay_alpha: float = 0.35             # bounding box fill transparency

    # ── HSV color ranges (tuned for indoor + daylight conditions) ─────────────
    # Format: color_name -> list of (lower, upper) ranges
    # Multiple ranges handle hue wrap-around (e.g., red straddles 0°/180°)
    color_ranges: Dict[str, List[HSVRange]] = field(default_factory=lambda: {
        "Red": [
            ((0,   90,  60), (10,  255, 255)),
            ((165, 90,  60), (180, 255, 255)),
        ],
        "Orange": [
            ((11, 100, 100), (24, 255, 255)),
        ],
        "Yellow": [
            ((25,  80,  80), (34, 255, 255)),
        ],
        "Green": [
            ((35,  50,  50), (85, 255, 255)),
        ],
        "Cyan": [
            ((86,  50,  50), (99, 255, 255)),
        ],
        "Blue": [
            ((100, 60,  50), (130, 255, 255)),
        ],
        "Violet": [
            ((131, 50,  50), (145, 255, 255)),
        ],
        "Magenta": [
            ((146, 60,  60), (164, 255, 255)),
        ],
        "White": [
            ((0, 0, 200), (180, 30, 255)),
        ],
        "Black": [
            ((0, 0, 0), (180, 255, 50)),
        ],
        "Gray": [
            ((0, 0, 51), (180, 29, 199)),
        ],
    })

    # ── Color-blind confusion pairs ───────────────────────────────────────────
    # Maps mode -> list of (color_a, color_b, warning_message)
    confusion_pairs: Dict[str, List[Tuple[str, str, str]]] = field(default_factory=lambda: {
        "deuteranopia": [
            ("Red",    "Green",  "⚠ Red & Green may look similar (Deuteranopia)"),
            ("Orange", "Green",  "⚠ Orange & Green may look similar (Deuteranopia)"),
            ("Red",    "Brown",  "⚠ Red & Brown may look similar (Deuteranopia)"),
        ],
        "protanopia": [
            ("Red",    "Green",  "⚠ Red & Green may look similar (Protanopia)"),
            ("Red",    "Black",  "⚠ Red & Black may look similar (Protanopia)"),
            ("Orange", "Yellow", "⚠ Orange & Yellow may look similar (Protanopia)"),
        ],
        "tritanopia": [
            ("Blue",   "Green",  "⚠ Blue & Green may look similar (Tritanopia)"),
            ("Blue",   "Gray",   "⚠ Blue & Gray may look similar (Tritanopia)"),
            ("Yellow", "White",  "⚠ Yellow & White may look similar (Tritanopia)"),
        ],
        "normal": [],
    })

    # ── BGR overlay colors for bounding boxes ─────────────────────────────────
    label_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "Red":     (0,   0,   220),
        "Orange":  (0,   128, 255),
        "Yellow":  (0,   220, 220),
        "Green":   (0,   200, 0),
        "Cyan":    (200, 200, 0),
        "Blue":    (220, 0,   0),
        "Violet":  (200, 0,   200),
        "Magenta": (180, 0,   180),
        "White":   (220, 220, 220),
        "Black":   (50,  50,  50),
        "Gray":    (140, 140, 140),
    })
