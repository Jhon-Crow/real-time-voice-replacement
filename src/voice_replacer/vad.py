"""
Voice Activity Detection (VAD) module using Silero VAD.
"""
import logging
import threading
from typing import Optional, Callable, List, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # ~32ms at 16kHz


class VoiceActivityDetector:
    """
    Voice Activity Detector using Silero VAD.

    Detects speech segments in real-time audio streams.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 50,
        sample_rate: int = 16000
    ):
        """
        Initialize VAD.

        Args:
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration to trigger
            min_silence_duration_ms: Minimum silence to end speech segment
            speech_pad_ms: Padding around speech segments
            sample_rate: Audio sample rate (8000 or 16000)
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate

        self._model = None
        self._model_utils = None
        self._lock = threading.Lock()
        self._initialized = False

        # State tracking
        self._is_speaking = False
        self._speech_buffer: List[np.ndarray] = []
        self._silence_samples = 0
        self._speech_samples = 0

        # Pre-speech buffer for padding
        self._pre_speech_buffer: deque = deque(
            maxlen=int(speech_pad_ms * sample_rate / 1000 / CHUNK_SIZE)
        )

    def initialize(self) -> bool:
        """
        Initialize the Silero VAD model.

        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._initialized:
                return True

            try:
                import torch
                torch.set_num_threads(1)  # Use single thread for low latency

                # Load Silero VAD
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )

                self._model = model
                self._model_utils = utils
                self._initialized = True
                logger.info("Silero VAD initialized successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize Silero VAD: {e}")
                return False

    def _get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """
        Get speech probability for an audio chunk.

        Args:
            audio_chunk: Audio data as float32 array

        Returns:
            Speech probability (0.0-1.0)
        """
        if not self._initialized or self._model is None:
            return 0.0

        try:
            import torch

            # Ensure correct format
            if len(audio_chunk) != CHUNK_SIZE:
                # Pad or truncate
                if len(audio_chunk) < CHUNK_SIZE:
                    audio_chunk = np.pad(
                        audio_chunk,
                        (0, CHUNK_SIZE - len(audio_chunk)),
                        mode='constant'
                    )
                else:
                    audio_chunk = audio_chunk[:CHUNK_SIZE]

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()

            # Get probability
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, self.sample_rate).item()

            return speech_prob

        except Exception as e:
            logger.error(f"Error in VAD inference: {e}")
            return 0.0

    def process_chunk(
        self,
        audio_chunk: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process an audio chunk and detect speech.

        Args:
            audio_chunk: Audio data as float32 array

        Returns:
            Tuple of (is_speaking, complete_speech_segment or None)
        """
        speech_prob = self._get_speech_prob(audio_chunk)
        is_speech = speech_prob >= self.threshold

        complete_segment = None
        samples_per_ms = self.sample_rate / 1000

        if is_speech:
            self._silence_samples = 0
            self._speech_samples += len(audio_chunk)

            if not self._is_speaking:
                # Check if we have enough speech to trigger
                if self._speech_samples >= self.min_speech_duration_ms * samples_per_ms:
                    self._is_speaking = True
                    # Add pre-speech buffer
                    for pre_chunk in self._pre_speech_buffer:
                        self._speech_buffer.append(pre_chunk)
                    logger.debug("Speech started")

            if self._is_speaking:
                self._speech_buffer.append(audio_chunk.copy())
            else:
                # Not yet triggered, keep in pre-buffer
                self._pre_speech_buffer.append(audio_chunk.copy())

        else:
            self._speech_samples = 0

            if self._is_speaking:
                self._silence_samples += len(audio_chunk)
                self._speech_buffer.append(audio_chunk.copy())  # Include silence for padding

                # Check if silence is long enough to end speech
                if self._silence_samples >= self.min_silence_duration_ms * samples_per_ms:
                    # Speech segment complete
                    complete_segment = np.concatenate(self._speech_buffer)
                    self._speech_buffer = []
                    self._is_speaking = False
                    logger.debug(f"Speech ended, segment length: {len(complete_segment) / self.sample_rate:.2f}s")

            else:
                # Keep updating pre-speech buffer
                self._pre_speech_buffer.append(audio_chunk.copy())

        return self._is_speaking, complete_segment

    def reset(self) -> None:
        """Reset VAD state."""
        self._is_speaking = False
        self._speech_buffer = []
        self._silence_samples = 0
        self._speech_samples = 0
        self._pre_speech_buffer.clear()

        # Reset model state if applicable
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass

    def is_speaking(self) -> bool:
        """Check if currently in speech segment."""
        return self._is_speaking

    def get_current_buffer(self) -> Optional[np.ndarray]:
        """
        Get the current speech buffer without ending the segment.

        Returns:
            Current speech audio or None
        """
        if self._speech_buffer:
            return np.concatenate(self._speech_buffer)
        return None


class SimpleVAD:
    """
    Simple energy-based VAD as fallback when Silero is not available.
    """

    def __init__(
        self,
        threshold_db: float = -35.0,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        sample_rate: int = 16000
    ):
        """
        Initialize simple VAD.

        Args:
            threshold_db: Energy threshold in dB
            min_speech_duration_ms: Minimum speech duration
            min_silence_duration_ms: Minimum silence to end speech
            sample_rate: Audio sample rate
        """
        self.threshold_db = threshold_db
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate

        self._is_speaking = False
        self._speech_buffer: List[np.ndarray] = []
        self._silence_samples = 0
        self._speech_samples = 0
        self._initialized = True

    def initialize(self) -> bool:
        """Initialize (no-op for simple VAD)."""
        return True

    def _get_energy_db(self, audio_chunk: np.ndarray) -> float:
        """Calculate energy in dB."""
        rms = np.sqrt(np.mean(audio_chunk ** 2) + 1e-10)
        return 20 * np.log10(rms + 1e-10)

    def process_chunk(
        self,
        audio_chunk: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Process an audio chunk."""
        energy_db = self._get_energy_db(audio_chunk)
        is_speech = energy_db >= self.threshold_db

        complete_segment = None
        samples_per_ms = self.sample_rate / 1000

        if is_speech:
            self._silence_samples = 0
            self._speech_samples += len(audio_chunk)

            if not self._is_speaking:
                if self._speech_samples >= self.min_speech_duration_ms * samples_per_ms:
                    self._is_speaking = True

            if self._is_speaking:
                self._speech_buffer.append(audio_chunk.copy())

        else:
            self._speech_samples = 0

            if self._is_speaking:
                self._silence_samples += len(audio_chunk)
                self._speech_buffer.append(audio_chunk.copy())

                if self._silence_samples >= self.min_silence_duration_ms * samples_per_ms:
                    complete_segment = np.concatenate(self._speech_buffer)
                    self._speech_buffer = []
                    self._is_speaking = False

        return self._is_speaking, complete_segment

    def reset(self) -> None:
        """Reset VAD state."""
        self._is_speaking = False
        self._speech_buffer = []
        self._silence_samples = 0
        self._speech_samples = 0

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking


def create_vad(use_silero: bool = True, **kwargs) -> VoiceActivityDetector:
    """
    Factory function to create a VAD instance.

    Args:
        use_silero: Whether to use Silero VAD (falls back to simple if unavailable)
        **kwargs: Additional arguments for VAD

    Returns:
        VAD instance
    """
    if use_silero:
        vad = VoiceActivityDetector(**kwargs)
        if vad.initialize():
            return vad
        logger.warning("Silero VAD not available, falling back to simple VAD")

    return SimpleVAD(**kwargs)
