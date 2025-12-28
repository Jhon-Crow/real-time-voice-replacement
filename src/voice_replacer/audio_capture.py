"""
Audio capture module for real-time microphone input.
"""
import logging
import threading
import queue
from typing import Optional, Callable, List
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Real-time audio capture from microphone.

    Captures audio in chunks and provides them via a queue or callback.
    """

    def __init__(
        self,
        config: AudioConfig,
        device: Optional[str] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None
    ):
        """
        Initialize audio capture.

        Args:
            config: Audio configuration settings
            device: Input device name or index (None for default)
            callback: Optional callback function for each audio chunk
        """
        if sd is None:
            raise ImportError("sounddevice is required for audio capture. "
                            "Install with: pip install sounddevice")

        self.config = config
        self.device = device
        self.callback = callback
        self.audio_queue: queue.Queue = queue.Queue()

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status
    ) -> None:
        """Internal callback for sounddevice stream."""
        if status:
            logger.warning(f"Audio capture status: {status}")

        # Convert to mono float32 if needed
        audio_chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        audio_chunk = audio_chunk.astype(np.float32)

        # Put in queue for processing
        try:
            self.audio_queue.put_nowait(audio_chunk.copy())
        except queue.Full:
            logger.warning("Audio queue full, dropping chunk")

        # Call user callback if provided
        if self.callback is not None:
            try:
                self.callback(audio_chunk)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")

    def start(self) -> None:
        """Start audio capture."""
        with self._lock:
            if self._running:
                logger.warning("Audio capture already running")
                return

            try:
                self._stream = sd.InputStream(
                    device=self.device,
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype=self.config.dtype,
                    blocksize=self.config.chunk_size,
                    callback=self._audio_callback,
                    latency='low'
                )
                self._stream.start()
                self._running = True
                logger.info(f"Audio capture started (device: {self.device or 'default'})")
            except Exception as e:
                logger.error(f"Failed to start audio capture: {e}")
                raise

    def stop(self) -> None:
        """Stop audio capture."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            # Clear the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            logger.info("Audio capture stopped")

    def get_chunk(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the queue.

        Args:
            timeout: Timeout in seconds (None for non-blocking)

        Returns:
            Audio chunk as numpy array, or None if not available
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_running(self) -> bool:
        """Check if audio capture is running."""
        return self._running

    @staticmethod
    def list_devices() -> List[dict]:
        """
        List available audio input devices.

        Returns:
            List of device info dictionaries
        """
        if sd is None:
            return []

        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate'],
                    'is_default': device.get('default_input_device', False),
                })
        return devices

    @staticmethod
    def get_default_device() -> Optional[dict]:
        """Get the default input device."""
        if sd is None:
            return None

        try:
            device_info = sd.query_devices(kind='input')
            return {
                'index': sd.default.device[0],
                'name': device_info['name'],
                'channels': device_info['max_input_channels'],
                'sample_rate': device_info['default_samplerate'],
            }
        except Exception as e:
            logger.error(f"Error getting default device: {e}")
            return None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
