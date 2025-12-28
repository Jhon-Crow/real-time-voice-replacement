"""Tests for configuration module."""
import json
import tempfile
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from voice_replacer.config import (
    AppConfig, AudioConfig, ASRConfig, TTSConfig, OutputConfig
)


def test_default_config():
    """Test default configuration values."""
    config = AppConfig()

    assert config.audio.sample_rate == 16000
    assert config.audio.channels == 1
    assert config.audio.vad_threshold == 0.5

    assert config.asr.model_name == 'vosk-model-small-en-us-0.15'

    assert config.tts.model_name == 'en_US-lessac-medium'
    assert config.tts.speed == 1.0

    assert config.enabled == False
    assert config.use_gpu == False


def test_config_save_load():
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'

        # Create and modify config
        config = AppConfig()
        config.tts.speed = 1.5
        config.enabled = True
        config.input_device = 'Test Microphone'

        # Save
        config.save(config_path)

        # Verify file exists
        assert config_path.exists()

        # Load
        loaded = AppConfig.load(config_path)

        assert loaded.tts.speed == 1.5
        assert loaded.enabled == True
        assert loaded.input_device == 'Test Microphone'


def test_audio_config():
    """Test audio configuration."""
    config = AudioConfig(
        sample_rate=22050,
        channels=2,
        chunk_size=1024
    )

    assert config.sample_rate == 22050
    assert config.channels == 2
    assert config.chunk_size == 1024


def test_config_json_format():
    """Test configuration JSON format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'

        config = AppConfig()
        config.save(config_path)

        with open(config_path, 'r') as f:
            data = json.load(f)

        assert 'audio' in data
        assert 'asr' in data
        assert 'tts' in data
        assert 'output' in data
        assert 'enabled' in data


def test_config_load_nonexistent():
    """Test loading from non-existent file returns defaults."""
    config = AppConfig.load(Path('/nonexistent/config.json'))

    # Should return defaults
    assert config.audio.sample_rate == 16000
    assert config.enabled == False
