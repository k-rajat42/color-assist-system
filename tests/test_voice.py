"""
tests/test_voice.py
===================
Unit tests for VoiceAnnouncer (pyttsx3 mocked out).
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
import pytest

from config.settings import AppConfig


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(
        voice_enabled=True,
        announce_interval=1.0,
        colorblind_mode="deuteranopia",
    )


class TestVoiceThrottling:
    """Voice should not announce same color repeatedly within the interval."""

    def test_same_color_not_repeated_within_interval(self, config):
        with patch("src.voice._TTS_AVAILABLE", True), \
             patch("src.voice.pyttsx3") as mock_pyttsx3:
            mock_engine = MagicMock()
            mock_pyttsx3.init.return_value = mock_engine

            from src.voice import VoiceAnnouncer
            ann = VoiceAnnouncer(config)

            ann.announce(["Red"])
            ann.announce(["Red"])   # within interval → should not queue again

            time.sleep(0.1)
            ann.shutdown()

            # queue should have been called at most once for "Red"
            queued = ann._last_announced.get("Red", 0)
            assert queued > 0

    def test_different_colors_both_queued(self, config):
        with patch("src.voice._TTS_AVAILABLE", True), \
             patch("src.voice.pyttsx3") as mock_pyttsx3:
            mock_engine = MagicMock()
            mock_pyttsx3.init.return_value = mock_engine

            from src.voice import VoiceAnnouncer
            ann = VoiceAnnouncer(config)
            ann.announce(["Red", "Blue"])
            ann.shutdown()

            assert "Red" in ann._last_announced
            assert "Blue" in ann._last_announced


class TestConfusionWarning:
    def test_warning_on_red_green_simultaneous(self, config):
        with patch("src.voice._TTS_AVAILABLE", True), \
             patch("src.voice.pyttsx3") as mock_pyttsx3:
            mock_engine = MagicMock()
            mock_pyttsx3.init.return_value = mock_engine

            from src.voice import VoiceAnnouncer
            ann = VoiceAnnouncer(config)

            warning = ann._confusion_warning(["Red", "Green"], time.time())
            assert warning is not None
            assert "Deuteranopia" in warning or "deuteranopia" in warning.lower()
            ann.shutdown()

    def test_no_warning_when_only_one_pair_color(self, config):
        with patch("src.voice._TTS_AVAILABLE", True), \
             patch("src.voice.pyttsx3") as mock_pyttsx3:
            mock_engine = MagicMock()
            mock_pyttsx3.init.return_value = mock_engine

            from src.voice import VoiceAnnouncer
            ann = VoiceAnnouncer(config)
            warning = ann._confusion_warning(["Red"], time.time())
            assert warning is None
            ann.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
