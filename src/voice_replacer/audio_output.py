"""
Audio output module for virtual microphone output.

Supports output to VB-Audio Virtual Cable or other virtual audio devices.
"""
import logging
import queue
import threading
from typing import Optional, List, Callable
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

logger = logging.getLogger(__name__)


class AudioOutput:
    """
    Real-time audio output to a virtual microphone.

    Outputs synthesized audio to a virtual audio device (e.g., VB-Audio Cable)
    so other applications can use it as microphone input.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 22050,
        channels: int = 1,
        buffer_size: int = 2048,
        dtype: str = 'float32'
    ):
        """
        Initialize audio output.

        Args:
            device: Output device name or index (None for default)
            sample_rate: Output sample rate
            channels: Number of channels
            buffer_size: Buffer size for output
            dtype: Audio data type
        """
        if sd is None:
            raise ImportError("sounddevice is required. Install with: pip install sounddevice")

        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.dtype = dtype

        self._stream: Optional[sd.OutputStream] = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._running = False
        self._lock = threading.Lock()

        # Current audio being played
        self._current_audio: Optional[np.ndarray] = None
        self._current_position = 0

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status
    ) -> None:
        """Internal callback for audio output."""
        if status:
            logger.warning(f"Audio output status: {status}")

        output = np.zeros((frames, self.channels), dtype=self.dtype)

        # Fill from current audio or queue
        written = 0
        while written < frames:
            # Get audio from current buffer or queue
            if self._current_audio is None or self._current_position >= len(self._current_audio):
                try:
                    self._current_audio = self._audio_queue.get_nowait()
                    self._current_position = 0
                except queue.Empty:
                    # No more audio, output silence
                    break

            # Copy available samples
            available = len(self._current_audio) - self._current_position
            to_copy = min(available, frames - written)

            if self.channels == 1:
                output[written:written + to_copy, 0] = \
                    self._current_audio[self._current_position:self._current_position + to_copy]
            else:
                # Duplicate mono to stereo
                for c in range(self.channels):
                    output[written:written + to_copy, c] = \
                        self._current_audio[self._current_position:self._current_position + to_copy]

            written += to_copy
            self._current_position += to_copy

        outdata[:] = output

    def start(self) -> None:
        """Start audio output stream."""
        with self._lock:
            if self._running:
                logger.warning("Audio output already running")
                return

            try:
                self._stream = sd.OutputStream(
                    device=self.device,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    blocksize=self.buffer_size,
                    callback=self._audio_callback,
                    latency='low'
                )
                self._stream.start()
                self._running = True
                logger.info(f"Audio output started (device: {self.device or 'default'})")
            except Exception as e:
                logger.error(f"Failed to start audio output: {e}")
                raise

    def stop(self) -> None:
        """Stop audio output stream."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            # Clear queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            self._current_audio = None
            self._current_position = 0

            logger.info("Audio output stopped")

    def play(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Queue audio for playback.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of audio (if different from output)
        """
        if not self._running:
            logger.warning("Audio output not running")
            return

        if len(audio) == 0:
            return

        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)

        # Ensure float32
        audio = audio.astype(np.float32)

        try:
            self._audio_queue.put(audio, timeout=1.0)
        except queue.Full:
            logger.warning("Audio output queue full, dropping audio")

    def play_blocking(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play audio and block until complete.

        Args:
            audio: Audio data
            sample_rate: Sample rate
        """
        if len(audio) == 0:
            return

        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)

        # Use sounddevice.play for blocking playback
        try:
            sd.play(audio, self.sample_rate, device=self.device)
            sd.wait()
        except Exception as e:
            logger.error(f"Blocking playback error: {e}")

    def _resample(
        self,
        audio: np.ndarray,
        src_rate: int,
        dst_rate: int
    ) -> np.ndarray:
        """
        Resample audio to different sample rate.

        Args:
            audio: Input audio
            src_rate: Source sample rate
            dst_rate: Destination sample rate

        Returns:
            Resampled audio
        """
        if src_rate == dst_rate:
            return audio

        try:
            from scipy import signal
            num_samples = int(len(audio) * dst_rate / src_rate)
            return signal.resample(audio, num_samples).astype(np.float32)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = dst_rate / src_rate
            new_length = int(len(audio) * ratio)
            indices = np.arange(new_length) / ratio
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def clear_queue(self) -> None:
        """Clear any pending audio in the queue."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._current_audio = None
        self._current_position = 0

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return (
            self._running and
            (not self._audio_queue.empty() or
             (self._current_audio is not None and
              self._current_position < len(self._current_audio)))
        )

    def is_running(self) -> bool:
        """Check if output stream is running."""
        return self._running

    def set_device(self, device: Optional[str]) -> None:
        """
        Change output device.

        Args:
            device: New device name or index
        """
        was_running = self._running
        if was_running:
            self.stop()

        self.device = device

        if was_running:
            self.start()

    def set_sample_rate(self, sample_rate: int) -> None:
        """
        Change output sample rate.

        Args:
            sample_rate: New sample rate
        """
        was_running = self._running
        if was_running:
            self.stop()

        self.sample_rate = sample_rate

        if was_running:
            self.start()

    @staticmethod
    def list_devices() -> List[dict]:
        """
        List available audio output devices.

        Returns:
            List of device info dictionaries
        """
        if sd is None:
            return []

        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_output_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_output_channels'],
                    'sample_rate': device['default_samplerate'],
                    'is_default': device.get('default_output_device', False),
                })
        return devices

    @staticmethod
    def find_virtual_cable() -> Optional[dict]:
        """
        Find VB-Audio Virtual Cable device.

        Returns:
            Device info or None if not found
        """
        devices = AudioOutput.list_devices()
        for device in devices:
            name_lower = device['name'].lower()
            if 'vb-audio' in name_lower or 'virtual cable' in name_lower:
                logger.info(f"Found virtual cable: {device['name']}")
                return device

            # Also check for common virtual audio device names
            if 'voicemeeter' in name_lower:
                logger.info(f"Found VoiceMeeter: {device['name']}")
                return device

        return None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
