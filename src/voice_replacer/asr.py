"""
Automatic Speech Recognition (ASR) module using Vosk.
"""
import json
import logging
import os
import threading
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import numpy as np

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    Offline speech recognition using Vosk.
    """

    # Available models
    MODELS = {
        'en-us-small': {
            'name': 'vosk-model-small-en-us-0.15',
            'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
            'size_mb': 40,
            'description': 'Small English (US) model - fast, good accuracy',
        },
        'en-us-large': {
            'name': 'vosk-model-en-us-0.22',
            'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
            'size_mb': 128,
            'description': 'Large English (US) model - slower, best accuracy',
        },
        'en-gb-small': {
            'name': 'vosk-model-small-en-gb-0.15',
            'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-gb-0.15.zip',
            'size_mb': 40,
            'description': 'Small English (UK) model',
        },
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = 'en-us-small',
        sample_rate: int = 16000,
        models_dir: Optional[Path] = None
    ):
        """
        Initialize speech recognizer.

        Args:
            model_path: Path to Vosk model directory
            model_name: Name of model to use (if model_path not specified)
            sample_rate: Audio sample rate
            models_dir: Directory to store downloaded models
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.models_dir = models_dir or Path.home() / '.vosk' / 'models'

        self._model = None
        self._recognizer = None
        self._lock = threading.Lock()
        self._initialized = False

        # Determine model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            model_info = self.MODELS.get(model_name, self.MODELS['en-us-small'])
            self.model_path = self.models_dir / model_info['name']

    def _download_model(self, progress_callback=None) -> bool:
        """
        Download the Vosk model if not present.

        Args:
            progress_callback: Optional callback(downloaded, total) for progress

        Returns:
            True if model is available
        """
        if self.model_path.exists():
            return True

        model_info = self.MODELS.get(self.model_name)
        if not model_info:
            logger.error(f"Unknown model: {self.model_name}")
            return False

        url = model_info['url']
        zip_path = self.models_dir / f"{model_info['name']}.zip"

        try:
            # Create models directory
            self.models_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading Vosk model from {url}...")

            def reporthook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if progress_callback:
                    progress_callback(downloaded, total_size)
                if block_num % 100 == 0:
                    percent = min(100, downloaded * 100 // total_size)
                    logger.info(f"Download progress: {percent}%")

            urllib.request.urlretrieve(url, zip_path, reporthook)

            # Extract
            logger.info("Extracting model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.models_dir)

            # Remove zip file
            zip_path.unlink()

            logger.info(f"Model downloaded to {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return False

    def initialize(self, progress_callback=None) -> bool:
        """
        Initialize the speech recognizer.

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
                if not self._download_model(progress_callback):
                    return False

                # Import Vosk
                from vosk import Model, KaldiRecognizer, SetLogLevel

                # Reduce Vosk logging
                SetLogLevel(-1)

                # Load model
                logger.info(f"Loading Vosk model from {self.model_path}...")
                self._model = Model(str(self.model_path))

                # Create recognizer
                self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
                self._recognizer.SetWords(True)

                self._initialized = True
                logger.info("Speech recognizer initialized successfully")
                return True

            except ImportError:
                logger.error("Vosk not installed. Install with: pip install vosk")
                return False
            except Exception as e:
                logger.error(f"Failed to initialize speech recognizer: {e}")
                return False

    def recognize(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Recognize speech from audio.

        Args:
            audio: Audio data as float32 array (16kHz mono)

        Returns:
            Tuple of (recognized_text, confidence)
        """
        if not self._initialized:
            logger.warning("Speech recognizer not initialized")
            return "", 0.0

        try:
            # Convert to int16 bytes as required by Vosk
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            # Process audio
            self._recognizer.AcceptWaveform(audio_bytes)

            # Get result
            result = json.loads(self._recognizer.FinalResult())

            text = result.get('text', '').strip()

            # Calculate confidence from word-level results
            confidence = 0.0
            if 'result' in result and result['result']:
                confidences = [w.get('conf', 0.0) for w in result['result']]
                confidence = sum(confidences) / len(confidences)

            return text, confidence

        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return "", 0.0

    def recognize_streaming(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Process audio chunk for streaming recognition.

        Args:
            audio_chunk: Audio chunk as float32 array

        Returns:
            Partial result if available, None otherwise
        """
        if not self._initialized:
            return None

        try:
            # Convert to int16 bytes
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            if self._recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self._recognizer.Result())
                text = result.get('text', '').strip()
                if text:
                    return text

            return None

        except Exception as e:
            logger.error(f"Streaming recognition error: {e}")
            return None

    def get_partial_result(self) -> str:
        """
        Get partial recognition result.

        Returns:
            Partial recognized text
        """
        if not self._initialized:
            return ""

        try:
            result = json.loads(self._recognizer.PartialResult())
            return result.get('partial', '')
        except Exception:
            return ""

    def reset(self) -> None:
        """Reset recognizer state."""
        if self._recognizer is not None:
            try:
                # Get final result to clear state
                self._recognizer.FinalResult()
            except Exception:
                pass

    def is_initialized(self) -> bool:
        """Check if recognizer is initialized."""
        return self._initialized

    @classmethod
    def list_models(cls) -> dict:
        """List available models."""
        return cls.MODELS.copy()
