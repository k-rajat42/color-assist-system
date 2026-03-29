"""
src/voice.py
============
Threaded TTS announcer using pyttsx3.

Features
--------
- Non-blocking: speech runs in a daemon thread; the main loop never waits.
- Throttling: each color announced no more than once per `announce_interval` seconds.
- Color-blind warnings: emits contextual confusion warnings for the configured mode.
- Graceful degradation: if pyttsx3 is unavailable, logs instead of crashing.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    logger.warning("pyttsx3 not installed — voice output disabled.")

from config.settings import AppConfig


class VoiceAnnouncer:
    """
    Thread-safe, rate-limited TTS wrapper.

    Usage
    -----
    announcer = VoiceAnnouncer(config)
    announcer.announce(["Red", "Green"])   # call every frame; internally throttled
    announcer.shutdown()
    """

    def __init__(self, config: AppConfig) -> None:
        self._enabled = config.voice_enabled and _TTS_AVAILABLE
        self._interval = config.announce_interval
        self._mode = config.colorblind_mode
        self._confusion_pairs: List[Tuple[str, str, str]] = (
            config.confusion_pairs.get(self._mode, [])
        )

        self._last_announced: Dict[str, float] = {}
        self._last_warning_time: float = 0.0
        self._warning_cooldown: float = 8.0   # seconds between same warning

        self._queue: List[str] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        if self._enabled:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   config.tts_rate)
            self._engine.setProperty("volume", config.tts_volume)
            self._worker = threading.Thread(
                target=self._speak_loop, daemon=True, name="TTS-Worker"
            )
            self._worker.start()
            logger.info("Voice announcer started (rate=%d wpm).", config.tts_rate)
        else:
            logger.info("Voice announcer disabled.")

    # ── Public API ────────────────────────────────────────────────────────────

    def announce(self, detected_colors: List[str]) -> None:
        """
        Enqueue announcements for newly detected colors.
        Call once per frame — internally rate-limited per color.
        """
        if not self._enabled or not detected_colors:
            return

        now = time.time()
        to_speak: List[str] = []

        for color in detected_colors:
            last = self._last_announced.get(color, 0.0)
            if now - last >= self._interval:
                to_speak.append(color)
                self._last_announced[color] = now

        # Check confusion warnings
        warning = self._confusion_warning(detected_colors, now)

        with self._lock:
            if to_speak:
                phrase = "Detected: " + ", ".join(to_speak)
                self._queue.append(phrase)
                logger.info("Queued announcement: '%s'", phrase)
            if warning:
                self._queue.append(warning)
                self._last_warning_time = now

    def shutdown(self) -> None:
        """Stop the TTS worker thread."""
        self._stop_event.set()
        if self._enabled:
            self._worker.join(timeout=3.0)
        logger.debug("VoiceAnnouncer shutdown complete.")

    # ── Private ───────────────────────────────────────────────────────────────

    def _confusion_warning(
        self, detected: List[str], now: float
    ) -> Optional[str]:
        """Return a warning string if a confusion pair is simultaneously detected."""
        if now - self._last_warning_time < self._warning_cooldown:
            return None
        detected_set: Set[str] = set(detected)
        for a, b, message in self._confusion_pairs:
            if a in detected_set and b in detected_set:
                logger.warning(message)
                return message
        return None

    def _speak_loop(self) -> None:
        """Worker: drain the speech queue, one utterance at a time."""
        while not self._stop_event.is_set():
            phrase: Optional[str] = None
            with self._lock:
                if self._queue:
                    phrase = self._queue.pop(0)

            if phrase:
                try:
                    self._engine.say(phrase)
                    self._engine.runAndWait()
                except Exception as exc:
                    logger.error("TTS error: %s", exc)
            else:
                time.sleep(0.05)
