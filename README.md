# 🎨 Color Assist System for Color-Blind Users

> Real-time webcam color detection with voice guidance and color-blindness assistance warnings.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Project Architecture](#project-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [How It Works](#how-it-works)
8. [Color Blindness Modes](#color-blindness-modes)
9. [Running Tests](#running-tests)
10. [Troubleshooting](#troubleshooting)
11. [Future Improvements](#future-improvements)

---

## Overview

Color Assist System is a Python-based Computer Vision application that helps users with color vision deficiency (color blindness) identify colors in real time using a standard webcam.  
The system announces detected colors through voice output and warns users when two colors that are easily confused in their specific type of color blindness appear simultaneously on screen.

**Technology stack:** Python 3.9+ · OpenCV 4.8+ · NumPy · pyttsx3 · pytest

---

## Features

| Feature | Description |
|---|---|
| **Real-time detection** | Processes live webcam feed at up to 30 FPS |
| **HSV color segmentation** | Detects 11 colors: Red, Orange, Yellow, Green, Cyan, Blue, Violet, Magenta, White, Black, Gray |
| **Adaptive lighting** | CLAHE equalisation on the V-channel keeps detection stable indoors and outdoors |
| **Object tracking** | CSRT/KCF/MOSSE tracker smooths bounding boxes between detector frames |
| **Confidence scores** | Each detection carries a 0–100% confidence badge |
| **Voice output** | Non-blocking threaded TTS via pyttsx3 — main loop never stalls |
| **Color-blind warnings** | Mode-specific confusion-pair alerts (e.g., "Red & Green may look similar — Deuteranopia") |
| **CLI-only** | No GUI dependency; runs on headless servers via X-forwarding |
| **Modular code** | Each concern in its own module; fully unit-tested |

---

## Project Architecture

```
color_assist_system/
│
├── main.py                  ← CLI entry point & argument parsing
│
├── config/
│   ├── __init__.py
│   └── settings.py          ← All parameters in one dataclass (AppConfig)
│
├── src/
│   ├── __init__.py
│   ├── camera.py            ← VideoCapture wrapper (retry, configure, context manager)
│   ├── detector.py          ← HSV + CLAHE color segmentation → Detection objects
│   ├── tracker.py           ← OpenCV tracker per-color with auto re-init
│   ├── voice.py             ← Threaded pyttsx3 TTS with rate-limiting & warnings
│   ├── renderer.py          ← Bounding boxes, confidence badges, HUD overlay
│   └── pipeline.py          ← Orchestrates Camera → Detector → Tracker → Voice → Render
│
├── tests/
│   ├── test_detector.py     ← Detector unit tests (synthetic frames, no camera needed)
│   └── test_voice.py        ← Voice throttling & confusion-warning tests
│
├── logs/                    ← Session logs written here (when --save-log is used)
├── requirements.txt
└── README.md
```

### Data flow

```
Webcam frame
    │
    ▼
Camera.read()
    │
    ▼
ColorDetector.detect()          ← Gaussian blur → HSV → CLAHE → mask per color
    │                              → morphological clean → contours → Detection[]
    ▼
ObjectTracker.update()          ← CSRT tracker smooths each bbox
    │
    ├──► VoiceAnnouncer.announce()   ← Throttled TTS + confusion warnings
    │
    ▼
FrameRenderer.draw_frame()      ← Overlays + HUD
    │
    ▼
cv2.imshow()
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- A USB or built-in webcam
- A speaker / audio output (for voice)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/color-assist-system.git
cd color-assist-system

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the system
python main.py
```

---

## Usage

```
python main.py [OPTIONS]

Options:
  --source INT           Webcam index (default: 0)
  --mode CHOICE          Color blindness mode: deuteranopia | protanopia |
                         tritanopia | normal  (default: deuteranopia)
  --no-voice             Disable voice announcements
  --save-log             Save detection log to logs/session.log
  --confidence FLOAT     Minimum confidence threshold 0.0–1.0 (default: 0.65)
  --announce-interval S  Seconds between repeated announcements (default: 2.5)
```

### Examples

```bash
# Standard run — deuteranopia mode, voice on
python main.py

# Protanopia user, second camera, stricter confidence
python main.py --mode protanopia --source 1 --confidence 0.75

# Silent mode with session log
python main.py --no-voice --save-log

# Full debug for academic demo
python main.py --mode tritanopia --confidence 0.5 --announce-interval 1.5
```

**Press `Q`** in the display window to quit cleanly.

---

## Configuration

All tuneable parameters are in `config/settings.py`.  
Key parameters and their purpose:

| Parameter | Default | Effect |
|---|---|---|
| `confidence_threshold` | 0.65 | Filters weak detections; increase to reduce false positives |
| `min_contour_area` | 2500 px² | Minimum blob size; increase to ignore small noise |
| `clahe_clip_limit` | 2.0 | CLAHE contrast limit; increase for very dark environments |
| `announce_interval` | 2.5 s | How often the same color is re-announced |
| `tracker_type` | CSRT | CSRT = most accurate; KCF = faster; MOSSE = fastest |
| `tracker_reinit_interval` | 45 frames | How often the tracker is reset from fresh detections |
| `gaussian_blur_kernel` | (5, 5) | Increase to (9,9) for very noisy/grainy cameras |

**HSV ranges** can also be tuned directly in `AppConfig.color_ranges` to match your specific lighting environment.

---

## How It Works

### 1. Preprocessing
Each frame is resized to 640×360 for processing speed, then Gaussian-blurred to remove sensor noise.

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalisation)
The frame is converted to HSV.  
The **Value (V) channel** is equalised using CLAHE.  
This normalises brightness locally — the same orange looks the same whether you're in a bright lab or a dimly lit room.

### 3. HSV Masking
For each configured color, `cv2.inRange()` is applied with the lower/upper HSV bounds from `AppConfig`.  
Red needs two ranges (it straddles 0°/180° in the hue wheel).

### 4. Morphological Cleaning
- **Opening** (erode then dilate): removes isolated noise pixels
- **Closing** (dilate then erode): fills gaps inside blobs

### 5. Contour Extraction & Confidence Scoring
`cv2.findContours()` locates external contours.  
Each contour is scored:

```
confidence = 0.55 × fill_ratio + 0.45 × size_score

fill_ratio   = contour_area / bounding_box_area   (compactness)
size_score   = 1 − exp(−area / 5000)              (sigmoid-style, saturates at large blobs)
```

Only detections ≥ `confidence_threshold` are returned.

### 6. Object Tracking
The CSRT tracker is seeded from the highest-confidence detection of each color.  
It provides frame-to-frame smoothing — the bounding box doesn't jitter even when the detector is temporarily uncertain.  
The tracker is re-seeded every `tracker_reinit_interval` frames to prevent drift.

### 7. Voice Output
Announcements run in a daemon thread — `engine.runAndWait()` blocks that thread, not the main loop.  
Each color is rate-limited by `announce_interval`.  
Confusion warnings appear when both colors in a known pair are simultaneously detected.

---

## Color Blindness Modes

| Mode | Condition | Confused pairs |
|---|---|---|
| `deuteranopia` | Green-blind (most common, ~6% of males) | Red↔Green, Orange↔Green, Red↔Brown |
| `protanopia` | Red-blind (~2% of males) | Red↔Green, Red↔Black, Orange↔Yellow |
| `tritanopia` | Blue-Yellow blind (rare) | Blue↔Green, Blue↔Gray, Yellow↔White |
| `normal` | No assistance warnings | — |

Warnings appear as a banner at the bottom of the screen and are announced via voice.

---

## Running Tests

```bash
# All tests (no webcam needed)
pytest tests/ -v

# Individual module
pytest tests/test_detector.py -v
pytest tests/test_voice.py -v

# With coverage report
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

Tests use synthetic NumPy frames and mocked pyttsx3 — a webcam is **not required** to run them.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `RuntimeError: Cannot open camera source 0` | Check webcam is connected; try `--source 1` |
| `pyttsx3 not installed — voice disabled` | Run `pip install pyttsx3`; on Linux also install `espeak`: `sudo apt install espeak` |
| Too many false detections | Increase `--confidence` to 0.75–0.85 |
| Colors not detected under dim light | Lower `confidence_threshold` to 0.5 and increase `clahe_clip_limit` to 3.0 in settings |
| High CPU usage | Lower `frame_width`/`frame_height` in settings or use `--tracker-type MOSSE` |
| `cv2.TrackerCSRT_create` not found | Install `opencv-contrib-python` (not plain `opencv-python`) |

---

## Future Improvements

- [ ] Multi-object tracking (one tracker per *instance*, not per color)
- [ ] Color name localisation (Hindi, Spanish, etc.)
- [ ] Export annotated video to file (`--output video.mp4`)
- [ ] REST API mode for integration with mobile apps
- [ ] Deep-learning colour classifier (MobileNetV3) as an optional backend
- [ ] Colour-blindness simulation overlay to preview what a scene looks like through each lens

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

- OpenCV community for tracker implementations
- pyttsx3 for cross-platform TTS
- Color blindness research at Vischeck.com for confusion-pair data
