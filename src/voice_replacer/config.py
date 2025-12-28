"""
Configuration settings for the Voice Replacement System.
"""
import os
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_app_data_dir() -> Path:
    """Get the application data directory."""
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')))
    else:  # Linux/Mac (for development)
        base = Path(os.path.expanduser('~/.local/share'))

    app_dir = base / 'VoiceReplacer'
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_models_dir() -> Path:
    """Get the models directory."""
    models_dir = get_app_data_dir() / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000  # 16kHz for VAD and ASR
    channels: int = 1  # Mono
    chunk_size: int = 512  # ~32ms at 16kHz
    dtype: str = 'float32'

    # VAD settings
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 50


@dataclass
class ASRConfig:
    """Speech recognition configuration."""
    model_name: str = 'vosk-model-small-en-us-0.15'
    model_url: str = 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip'
    model_size_mb: int = 40  # Approximate size


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""
    model_name: str = 'en_US-lessac-medium'
    speaker_id: int = 0
    speed: float = 1.0
    pitch_shift: float = 0.0  # In semitones


@dataclass
class OutputConfig:
    """Audio output configuration."""
    output_device: Optional[str] = None  # None = default, or "VB-Audio Virtual Cable"
    buffer_size: int = 2048


@dataclass
class AppConfig:
    """Main application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # General settings
    input_device: Optional[str] = None  # None = default microphone
    enabled: bool = False
    show_activity_indicator: bool = True
    minimize_to_tray: bool = True
    start_minimized: bool = False

    # Performance settings
    use_gpu: bool = False  # CPU-only by default for compatibility
    low_latency_mode: bool = True

    # Debug settings
    debug_mode: bool = False
    log_level: str = 'INFO'

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = get_app_data_dir() / 'config.json'

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'AppConfig':
        """Load configuration from file."""
        if path is None:
            path = get_app_data_dir() / 'config.json'

        if not path.exists():
            logger.info("No configuration file found, using defaults")
            return cls()

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Reconstruct nested dataclasses
            config = cls(
                audio=AudioConfig(**data.get('audio', {})),
                asr=ASRConfig(**data.get('asr', {})),
                tts=TTSConfig(**data.get('tts', {})),
                output=OutputConfig(**data.get('output', {})),
                input_device=data.get('input_device'),
                enabled=data.get('enabled', False),
                show_activity_indicator=data.get('show_activity_indicator', True),
                minimize_to_tray=data.get('minimize_to_tray', True),
                start_minimized=data.get('start_minimized', False),
                use_gpu=data.get('use_gpu', False),
                low_latency_mode=data.get('low_latency_mode', True),
                debug_mode=data.get('debug_mode', False),
                log_level=data.get('log_level', 'INFO'),
            )
            logger.info(f"Configuration loaded from {path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return cls()


# Available voice presets
VOICE_PRESETS = {
    'en_US-lessac-medium': {
        'name': 'Lessac (US English, Medium)',
        'description': 'Clear, professional American English voice',
        'gender': 'male',
        'language': 'en_US',
    },
    'en_US-amy-medium': {
        'name': 'Amy (US English, Medium)',
        'description': 'Friendly female American English voice',
        'gender': 'female',
        'language': 'en_US',
    },
    'en_GB-alan-medium': {
        'name': 'Alan (British English, Medium)',
        'description': 'British male voice with clear pronunciation',
        'gender': 'male',
        'language': 'en_GB',
    },
}
