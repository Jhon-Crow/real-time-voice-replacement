"""Tests for audio capture and output modules."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from voice_replacer.config import AudioConfig


class TestAudioCapture:
    """Tests for AudioCapture class."""

    def test_import(self):
        """Test AudioCapture can be imported."""
        from voice_replacer.audio_capture import AudioCapture
        assert AudioCapture is not None

    def test_list_devices(self):
        """Test listing devices (mock sounddevice)."""
        with patch('voice_replacer.audio_capture.sd') as mock_sd:
            mock_sd.query_devices.return_value = [
                {
                    'name': 'Test Microphone',
                    'max_input_channels': 2,
                    'max_output_channels': 0,
                    'default_samplerate': 44100.0,
                },
                {
                    'name': 'Test Speaker',
                    'max_input_channels': 0,
                    'max_output_channels': 2,
                    'default_samplerate': 48000.0,
                },
            ]

            from voice_replacer.audio_capture import AudioCapture

            devices = AudioCapture.list_devices()

            # Should only return input devices
            assert len(devices) == 1
            assert devices[0]['name'] == 'Test Microphone'
            assert devices[0]['channels'] == 2


class TestAudioOutput:
    """Tests for AudioOutput class."""

    def test_import(self):
        """Test AudioOutput can be imported."""
        from voice_replacer.audio_output import AudioOutput
        assert AudioOutput is not None

    def test_list_devices(self):
        """Test listing output devices (mock sounddevice)."""
        with patch('voice_replacer.audio_output.sd') as mock_sd:
            mock_sd.query_devices.return_value = [
                {
                    'name': 'Test Microphone',
                    'max_input_channels': 2,
                    'max_output_channels': 0,
                    'default_samplerate': 44100.0,
                },
                {
                    'name': 'Test Speaker',
                    'max_input_channels': 0,
                    'max_output_channels': 2,
                    'default_samplerate': 48000.0,
                },
            ]

            from voice_replacer.audio_output import AudioOutput

            devices = AudioOutput.list_devices()

            # Should only return output devices
            assert len(devices) == 1
            assert devices[0]['name'] == 'Test Speaker'
            assert devices[0]['channels'] == 2

    def test_find_virtual_cable(self):
        """Test finding VB-Audio Virtual Cable."""
        with patch('voice_replacer.audio_output.sd') as mock_sd:
            mock_sd.query_devices.return_value = [
                {
                    'name': 'Speakers',
                    'max_input_channels': 0,
                    'max_output_channels': 2,
                    'default_samplerate': 48000.0,
                },
                {
                    'name': 'VB-Audio Virtual Cable',
                    'max_input_channels': 0,
                    'max_output_channels': 2,
                    'default_samplerate': 48000.0,
                },
            ]

            from voice_replacer.audio_output import AudioOutput

            virtual_cable = AudioOutput.find_virtual_cable()

            assert virtual_cable is not None
            assert 'vb-audio' in virtual_cable['name'].lower()

    def test_resample(self):
        """Test audio resampling."""
        with patch('voice_replacer.audio_output.sd'):
            from voice_replacer.audio_output import AudioOutput

            output = AudioOutput()

            # Create test audio
            audio = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)

            # Resample from 16000 to 22050
            resampled = output._resample(audio, 16000, 22050)

            expected_length = int(1000 * 22050 / 16000)
            assert len(resampled) == expected_length

    def test_no_resample_same_rate(self):
        """Test no resampling when rates match."""
        with patch('voice_replacer.audio_output.sd'):
            from voice_replacer.audio_output import AudioOutput

            output = AudioOutput()

            audio = np.ones(1000, dtype=np.float32)

            resampled = output._resample(audio, 16000, 16000)

            assert len(resampled) == 1000
            np.testing.assert_array_equal(resampled, audio)
