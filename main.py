"""
Color Assist System for Color-Blind Users
==========================================
Entry point for the CLI-based real-time color detection system.

Usage:
    python main.py [--source 0] [--mode deuteranopia] [--no-voice] [--save-log]
"""

import argparse
import sys
import logging
from src.pipeline import ColorAssistPipeline
from config.settings import AppConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Color Assist System — Real-time color detection for color-blind users",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source", type=int, default=0,
        help="Webcam source index (default: 0)"
    )
    parser.add_argument(
        "--mode", type=str, default="deuteranopia",
        choices=["deuteranopia", "protanopia", "tritanopia", "normal"],
        help=(
            "Color blindness simulation mode:\n"
            "  deuteranopia  - Red-Green (green-blind) [default]\n"
            "  protanopia    - Red-Green (red-blind)\n"
            "  tritanopia    - Blue-Yellow blind\n"
            "  normal        - Full color mode"
        )
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Disable voice announcements"
    )
    parser.add_argument(
        "--save-log", action="store_true",
        help="Save detection log to logs/session.log"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.65,
        help="Minimum confidence threshold for detection (0.0–1.0, default: 0.65)"
    )
    parser.add_argument(
        "--announce-interval", type=float, default=2.5,
        help="Minimum seconds between voice announcements (default: 2.5)"
    )
    return parser.parse_args()


def setup_logging(save_log: bool) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if save_log:
        handlers.append(logging.FileHandler("logs/session.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.save_log)
    logger = logging.getLogger(__name__)

    config = AppConfig(
        camera_source=args.source,
        colorblind_mode=args.mode,
        voice_enabled=not args.no_voice,
        confidence_threshold=args.confidence,
        announce_interval=args.announce_interval,
    )

    logger.info("=" * 55)
    logger.info("   Color Assist System — Starting")
    logger.info(f"   Mode        : {config.colorblind_mode.upper()}")
    logger.info(f"   Voice       : {'ON' if config.voice_enabled else 'OFF'}")
    logger.info(f"   Camera      : /dev/video{config.camera_source}")
    logger.info(f"   Confidence  : {config.confidence_threshold:.0%}")
    logger.info("=" * 55)
    logger.info("Press  Q  in the display window to quit.")

    try:
        pipeline = ColorAssistPipeline(config)
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Session terminated by user (Ctrl+C).")
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        sys.exit(1)
    finally:
        logger.info("Color Assist System shut down cleanly.")


if __name__ == "__main__":
    main()
