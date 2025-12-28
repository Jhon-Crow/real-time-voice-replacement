"""Tests for Voice Activity Detection module."""
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from voice_replacer.vad import SimpleVAD, VoiceActivityDetector


class TestSimpleVAD:
    """Tests for SimpleVAD (energy-based)."""

    def test_init(self):
        """Test initialization."""
        vad = SimpleVAD()
        assert vad._initialized == True
        assert vad._is_speaking == False

    def test_silence_detection(self):
        """Test detection of silence."""
        vad = SimpleVAD(threshold_db=-30.0)

        # Create silent audio (very low amplitude)
        silence = np.zeros(512, dtype=np.float32)

        is_speaking, segment = vad.process_chunk(silence)

        assert is_speaking == False
        assert segment is None

    def test_speech_detection(self):
        """Test detection of speech-like audio."""
        vad = SimpleVAD(
            threshold_db=-30.0,
            min_speech_duration_ms=100,
            sample_rate=16000
        )

        # Create audio with significant amplitude
        speech = np.random.randn(512).astype(np.float32) * 0.3

        # Process enough chunks to trigger speech
        for _ in range(10):
            is_speaking, segment = vad.process_chunk(speech)

        assert is_speaking == True

    def test_speech_to_silence_transition(self):
        """Test transition from speech to silence returns segment."""
        vad = SimpleVAD(
            threshold_db=-30.0,
            min_speech_duration_ms=50,
            min_silence_duration_ms=100,
            sample_rate=16000
        )

        # Generate speech-like audio
        speech = np.random.randn(512).astype(np.float32) * 0.3
        silence = np.zeros(512, dtype=np.float32)

        # Feed speech chunks
        for _ in range(10):
            vad.process_chunk(speech)

        # Feed silence chunks
        segment = None
        for _ in range(10):
            _, seg = vad.process_chunk(silence)
            if seg is not None:
                segment = seg
                break

        assert segment is not None
        assert len(segment) > 0

    def test_reset(self):
        """Test reset clears state."""
        vad = SimpleVAD()

        # Feed some audio
        speech = np.random.randn(512).astype(np.float32) * 0.3
        for _ in range(10):
            vad.process_chunk(speech)

        assert vad._is_speaking == True

        # Reset
        vad.reset()

        assert vad._is_speaking == False
        assert len(vad._speech_buffer) == 0


class TestVoiceActivityDetector:
    """Tests for Silero VAD wrapper."""

    def test_init(self):
        """Test initialization (without loading model)."""
        vad = VoiceActivityDetector()
        assert vad._initialized == False

    def test_uninitialized_returns_zero(self):
        """Test uninitialized VAD returns 0 probability."""
        vad = VoiceActivityDetector()

        audio = np.random.randn(512).astype(np.float32)
        prob = vad._get_speech_prob(audio)

        assert prob == 0.0

    def test_process_uninitialized(self):
        """Test processing without initialization."""
        vad = VoiceActivityDetector()

        audio = np.random.randn(512).astype(np.float32)
        is_speaking, segment = vad.process_chunk(audio)

        # Should not detect speech without initialization
        assert is_speaking == False
        assert segment is None

    def test_parameters(self):
        """Test parameter storage."""
        vad = VoiceActivityDetector(
            threshold=0.7,
            min_speech_duration_ms=300,
            min_silence_duration_ms=400,
            speech_pad_ms=100,
            sample_rate=8000
        )

        assert vad.threshold == 0.7
        assert vad.min_speech_duration_ms == 300
        assert vad.min_silence_duration_ms == 400
        assert vad.speech_pad_ms == 100
        assert vad.sample_rate == 8000
