"""
Text-to-Speech (TTS) module using Piper.
"""
import io
import logging
import os
import subprocess
import threading
import wave
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import urllib.request
import tarfile
import numpy as np

logger = logging.getLogger(__name__)


class PiperTTS:
    """
    Text-to-Speech using Piper TTS.

    Piper is a fast, local neural text to speech system.
    """

    # Available voice models
    VOICES = {
        'en_US-lessac-medium': {
            'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx',
            'config_url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json',
            'description': 'US English male voice (medium quality)',
            'sample_rate': 22050,
        },
        'en_US-amy-medium': {
            'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx',
            'config_url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json',
            'description': 'US English female voice (medium quality)',
            'sample_rate': 22050,
        },
        'en_GB-alan-medium': {
            'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx',
            'config_url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json',
            'description': 'British English male voice (medium quality)',
            'sample_rate': 22050,
        },
        'en_US-libritts-high': {
            'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx',
            'config_url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx.json',
            'description': 'US English multi-speaker (high quality)',
            'sample_rate': 22050,
            'multi_speaker': True,
        },
    }

    def __init__(
        self,
        voice: str = 'en_US-lessac-medium',
        speaker_id: int = 0,
        speed: float = 1.0,
        models_dir: Optional[Path] = None
    ):
        """
        Initialize Piper TTS.

        Args:
            voice: Voice model name
            speaker_id: Speaker ID for multi-speaker models
            speed: Speech rate multiplier
            models_dir: Directory to store downloaded models
        """
        self.voice = voice
        self.speaker_id = speaker_id
        self.speed = speed
        self.models_dir = models_dir or Path.home() / '.piper' / 'voices'

        self._piper = None
        self._lock = threading.Lock()
        self._initialized = False
        self._sample_rate = 22050

    def _download_voice(self, progress_callback=None) -> bool:
        """
        Download voice model if not present.

        Args:
            progress_callback: Optional callback(downloaded, total)

        Returns:
            True if model is available
        """
        voice_info = self.VOICES.get(self.voice)
        if not voice_info:
            logger.error(f"Unknown voice: {self.voice}")
            return False

        model_path = self.models_dir / f"{self.voice}.onnx"
        config_path = self.models_dir / f"{self.voice}.onnx.json"

        if model_path.exists() and config_path.exists():
            return True

        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)

            # Download model
            logger.info(f"Downloading Piper voice model: {self.voice}...")

            def reporthook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if progress_callback:
                    progress_callback(downloaded, total_size)
                if block_num % 100 == 0:
                    percent = min(100, downloaded * 100 // max(1, total_size))
                    logger.info(f"Download progress: {percent}%")

            urllib.request.urlretrieve(
                voice_info['url'],
                model_path,
                reporthook
            )

            # Download config
            logger.info("Downloading voice config...")
            urllib.request.urlretrieve(
                voice_info['config_url'],
                config_path
            )

            logger.info(f"Voice model downloaded to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download voice model: {e}")
            return False

    def initialize(self, progress_callback=None) -> bool:
        """
        Initialize TTS engine.

        Args:
            progress_callback: Optional callback for download progress

        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._initialized:
                return True

            try:
                # Download model if needed
                if not self._download_voice(progress_callback):
                    return False

                # Try to import piper-tts
                try:
                    from piper import PiperVoice

                    model_path = self.models_dir / f"{self.voice}.onnx"
                    config_path = self.models_dir / f"{self.voice}.onnx.json"

                    self._piper = PiperVoice.load(
                        str(model_path),
                        config_path=str(config_path)
                    )

                    # Get sample rate from voice info
                    voice_info = self.VOICES.get(self.voice, {})
                    self._sample_rate = voice_info.get('sample_rate', 22050)

                    self._initialized = True
                    logger.info("Piper TTS initialized successfully")
                    return True

                except ImportError:
                    logger.warning("piper-tts not available, trying CLI fallback")
                    return self._init_cli_fallback()

            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                return False

    def _init_cli_fallback(self) -> bool:
        """Initialize using piper CLI as fallback."""
        try:
            # Check if piper CLI is available
            result = subprocess.run(
                ['piper', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self._piper = 'cli'
                self._initialized = True
                logger.info("Using Piper CLI for TTS")
                return True
            return False
        except FileNotFoundError:
            logger.error("Piper CLI not found")
            return False

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (audio_data as float32 array, sample_rate)
        """
        if not self._initialized:
            logger.warning("TTS not initialized")
            return np.array([], dtype=np.float32), self._sample_rate

        if not text.strip():
            return np.array([], dtype=np.float32), self._sample_rate

        try:
            with self._lock:
                if self._piper == 'cli':
                    return self._synthesize_cli(text)
                else:
                    return self._synthesize_python(text)

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return np.array([], dtype=np.float32), self._sample_rate

    def _synthesize_python(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using Python library."""
        import wave
        import io

        # Synthesize to WAV
        audio_buffer = io.BytesIO()

        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self._sample_rate)

            # Synthesize
            for audio_bytes in self._piper.synthesize_stream_raw(
                text,
                speaker_id=self.speaker_id,
                length_scale=1.0 / self.speed if self.speed > 0 else 1.0
            ):
                wav_file.writeframes(audio_bytes)

        # Read back as numpy array
        audio_buffer.seek(0)
        with wave.open(audio_buffer, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

        # Convert to float32
        audio = audio.astype(np.float32) / 32767.0

        return audio, self._sample_rate

    def _synthesize_cli(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using CLI."""
        model_path = self.models_dir / f"{self.voice}.onnx"

        # Run piper
        result = subprocess.run(
            [
                'piper',
                '--model', str(model_path),
                '--output-raw'
            ],
            input=text.encode('utf-8'),
            capture_output=True
        )

        if result.returncode != 0:
            logger.error(f"Piper CLI error: {result.stderr.decode()}")
            return np.array([], dtype=np.float32), self._sample_rate

        # Parse raw audio
        audio = np.frombuffer(result.stdout, dtype=np.int16)
        audio = audio.astype(np.float32) / 32767.0

        return audio, self._sample_rate

    def get_sample_rate(self) -> int:
        """Get the output sample rate."""
        return self._sample_rate

    def set_voice(self, voice: str) -> bool:
        """
        Change the voice model.

        Args:
            voice: Voice model name

        Returns:
            True if successful
        """
        if voice not in self.VOICES:
            logger.error(f"Unknown voice: {voice}")
            return False

        self.voice = voice
        self._initialized = False
        return self.initialize()

    def set_speed(self, speed: float) -> None:
        """Set speech rate multiplier."""
        self.speed = max(0.1, min(3.0, speed))

    def set_speaker(self, speaker_id: int) -> None:
        """Set speaker ID for multi-speaker models."""
        self.speaker_id = speaker_id

    def is_initialized(self) -> bool:
        """Check if TTS is initialized."""
        return self._initialized

    @classmethod
    def list_voices(cls) -> Dict[str, dict]:
        """List available voices."""
        return cls.VOICES.copy()


class SimpleTTS:
    """
    Simple TTS fallback using espeak/pyttsx3.
    """

    def __init__(self, speed: float = 1.0):
        """Initialize simple TTS."""
        self.speed = speed
        self._engine = None
        self._initialized = False
        self._sample_rate = 22050

    def initialize(self, progress_callback=None) -> bool:
        """Initialize TTS engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', int(150 * self.speed))
            self._initialized = True
            logger.info("Simple TTS (pyttsx3) initialized")
            return True
        except ImportError:
            logger.error("pyttsx3 not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize simple TTS: {e}")
            return False

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize speech (saves to temp file)."""
        if not self._initialized:
            return np.array([], dtype=np.float32), self._sample_rate

        try:
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name

            self._engine.save_to_file(text, temp_path)
            self._engine.runAndWait()

            # Read WAV file
            with wave.open(temp_path, 'rb') as wav_file:
                self._sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)

            os.unlink(temp_path)

            # Convert to float32
            audio = audio.astype(np.float32) / 32767.0
            return audio, self._sample_rate

        except Exception as e:
            logger.error(f"Simple TTS error: {e}")
            return np.array([], dtype=np.float32), self._sample_rate

    def get_sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    def is_initialized(self) -> bool:
        """Check if initialized."""
        return self._initialized


def create_tts(use_piper: bool = True, **kwargs):
    """
    Factory function to create TTS instance.

    Args:
        use_piper: Whether to use Piper TTS
        **kwargs: Additional arguments

    Returns:
        TTS instance
    """
    if use_piper:
        tts = PiperTTS(**kwargs)
        if tts.initialize():
            return tts
        logger.warning("Piper TTS not available, falling back to simple TTS")

    return SimpleTTS(**kwargs)
