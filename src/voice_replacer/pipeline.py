"""
Real-time voice replacement pipeline.

Connects audio capture, VAD, ASR, TTS, and audio output into a
complete voice replacement system.
"""
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import numpy as np

from .config import AppConfig, AudioConfig
from .audio_capture import AudioCapture
from .audio_output import AudioOutput
from .vad import VoiceActivityDetector, SimpleVAD, create_vad
from .asr import SpeechRecognizer
from .tts import PiperTTS, create_tts

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline state."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PROCESSING = "processing"  # Currently processing speech
    ERROR = "error"


@dataclass
class PipelineStatus:
    """Current pipeline status."""
    state: PipelineState
    is_speaking: bool = False
    is_processing: bool = False
    last_text: str = ""
    error_message: str = ""
    latency_ms: float = 0.0


class VoiceReplacementPipeline:
    """
    Main voice replacement pipeline.

    Flow:
    1. Capture audio from microphone
    2. Detect speech using VAD
    3. When speech ends, transcribe using ASR
    4. Synthesize new voice using TTS
    5. Output to virtual microphone
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Application configuration
        """
        self.config = config or AppConfig()

        # Components
        self._audio_capture: Optional[AudioCapture] = None
        self._audio_output: Optional[AudioOutput] = None
        self._vad: Optional[VoiceActivityDetector] = None
        self._asr: Optional[SpeechRecognizer] = None
        self._tts: Optional[PiperTTS] = None

        # State
        self._state = PipelineState.STOPPED
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._process_thread: Optional[threading.Thread] = None

        # Status
        self._status = PipelineStatus(state=PipelineState.STOPPED)

        # Callbacks
        self._on_status_change: Optional[Callable[[PipelineStatus], None]] = None
        self._on_text_recognized: Optional[Callable[[str], None]] = None
        self._on_speech_synthesized: Optional[Callable[[np.ndarray], None]] = None

    def set_status_callback(
        self,
        callback: Callable[[PipelineStatus], None]
    ) -> None:
        """Set callback for status changes."""
        self._on_status_change = callback

    def set_text_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for recognized text."""
        self._on_text_recognized = callback

    def set_synthesis_callback(
        self,
        callback: Callable[[np.ndarray], None]
    ) -> None:
        """Set callback for synthesized audio."""
        self._on_speech_synthesized = callback

    def _update_status(
        self,
        state: Optional[PipelineState] = None,
        **kwargs
    ) -> None:
        """Update pipeline status."""
        if state is not None:
            self._status.state = state

        for key, value in kwargs.items():
            if hasattr(self._status, key):
                setattr(self._status, key, value)

        if self._on_status_change:
            try:
                self._on_status_change(self._status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def initialize(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """
        Initialize all pipeline components.

        Args:
            progress_callback: Optional callback(component_name, progress)

        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._state != PipelineState.STOPPED:
                logger.warning("Pipeline already initialized or running")
                return True

            self._update_status(state=PipelineState.INITIALIZING)

            try:
                # Initialize audio capture
                if progress_callback:
                    progress_callback("Audio Capture", 0.0)

                self._audio_capture = AudioCapture(
                    config=self.config.audio,
                    device=self.config.input_device
                )

                # Initialize VAD
                if progress_callback:
                    progress_callback("Voice Activity Detection", 0.2)

                self._vad = create_vad(
                    use_silero=True,
                    threshold=self.config.audio.vad_threshold,
                    min_speech_duration_ms=self.config.audio.min_speech_duration_ms,
                    min_silence_duration_ms=self.config.audio.min_silence_duration_ms,
                    speech_pad_ms=self.config.audio.speech_pad_ms,
                    sample_rate=self.config.audio.sample_rate
                )

                # Initialize ASR
                if progress_callback:
                    progress_callback("Speech Recognition", 0.4)

                self._asr = SpeechRecognizer(
                    model_name='en-us-small',
                    sample_rate=self.config.audio.sample_rate
                )

                def asr_progress(downloaded, total):
                    if progress_callback and total > 0:
                        progress_callback(
                            "Speech Recognition",
                            0.4 + 0.2 * (downloaded / total)
                        )

                if not self._asr.initialize(asr_progress):
                    raise RuntimeError("Failed to initialize ASR")

                # Initialize TTS
                if progress_callback:
                    progress_callback("Text-to-Speech", 0.6)

                self._tts = create_tts(
                    use_piper=True,
                    voice=self.config.tts.model_name,
                    speaker_id=self.config.tts.speaker_id,
                    speed=self.config.tts.speed
                )

                def tts_progress(downloaded, total):
                    if progress_callback and total > 0:
                        progress_callback(
                            "Text-to-Speech",
                            0.6 + 0.2 * (downloaded / total)
                        )

                if not self._tts.initialize(tts_progress):
                    raise RuntimeError("Failed to initialize TTS")

                # Initialize audio output
                if progress_callback:
                    progress_callback("Audio Output", 0.8)

                # Try to find virtual cable, otherwise use default
                virtual_cable = AudioOutput.find_virtual_cable()
                output_device = None
                if virtual_cable:
                    output_device = virtual_cable['index']
                elif self.config.output.output_device:
                    output_device = self.config.output.output_device

                self._audio_output = AudioOutput(
                    device=output_device,
                    sample_rate=self._tts.get_sample_rate(),
                    buffer_size=self.config.output.buffer_size
                )

                if progress_callback:
                    progress_callback("Complete", 1.0)

                logger.info("Pipeline initialized successfully")
                self._update_status(state=PipelineState.STOPPED)
                return True

            except Exception as e:
                logger.error(f"Pipeline initialization failed: {e}")
                self._update_status(
                    state=PipelineState.ERROR,
                    error_message=str(e)
                )
                return False

    def start(self) -> bool:
        """
        Start the voice replacement pipeline.

        Returns:
            True if started successfully
        """
        with self._lock:
            if self._state == PipelineState.RUNNING:
                logger.warning("Pipeline already running")
                return True

            if self._audio_capture is None:
                logger.error("Pipeline not initialized")
                return False

            try:
                # Start audio capture
                self._audio_capture.start()

                # Start audio output
                self._audio_output.start()

                # Start processing thread
                self._stop_event.clear()
                self._process_thread = threading.Thread(
                    target=self._process_loop,
                    daemon=True
                )
                self._process_thread.start()

                self._state = PipelineState.RUNNING
                self._update_status(state=PipelineState.RUNNING)
                logger.info("Pipeline started")
                return True

            except Exception as e:
                logger.error(f"Failed to start pipeline: {e}")
                self._update_status(
                    state=PipelineState.ERROR,
                    error_message=str(e)
                )
                return False

    def stop(self) -> None:
        """Stop the pipeline."""
        with self._lock:
            if self._state == PipelineState.STOPPED:
                return

            # Signal stop
            self._stop_event.set()

            # Wait for processing thread
            if self._process_thread is not None:
                self._process_thread.join(timeout=2.0)
                self._process_thread = None

            # Stop components
            if self._audio_capture is not None:
                self._audio_capture.stop()

            if self._audio_output is not None:
                self._audio_output.stop()

            # Reset VAD state
            if self._vad is not None:
                self._vad.reset()

            # Reset ASR state
            if self._asr is not None:
                self._asr.reset()

            self._state = PipelineState.STOPPED
            self._update_status(state=PipelineState.STOPPED)
            logger.info("Pipeline stopped")

    def _process_loop(self) -> None:
        """Main processing loop."""
        logger.info("Processing loop started")

        while not self._stop_event.is_set():
            try:
                # Get audio chunk from capture
                audio_chunk = self._audio_capture.get_chunk(timeout=0.1)

                if audio_chunk is None:
                    continue

                # Process through VAD
                is_speaking, speech_segment = self._vad.process_chunk(audio_chunk)

                # Update speaking status
                if is_speaking != self._status.is_speaking:
                    self._update_status(is_speaking=is_speaking)

                # If we have a complete speech segment, process it
                if speech_segment is not None:
                    self._process_speech_segment(speech_segment)

            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)

        logger.info("Processing loop ended")

    def _process_speech_segment(self, audio: np.ndarray) -> None:
        """
        Process a complete speech segment.

        Args:
            audio: Speech audio as float32 array
        """
        start_time = time.time()
        self._update_status(is_processing=True)

        try:
            # Transcribe speech
            text, confidence = self._asr.recognize(audio)

            if not text:
                logger.debug("No text recognized")
                self._update_status(is_processing=False)
                return

            logger.info(f"Recognized: '{text}' (confidence: {confidence:.2f})")
            self._update_status(last_text=text)

            if self._on_text_recognized:
                self._on_text_recognized(text)

            # Synthesize new voice
            synth_audio, sample_rate = self._tts.synthesize(text)

            if len(synth_audio) == 0:
                logger.warning("TTS produced no audio")
                self._update_status(is_processing=False)
                return

            if self._on_speech_synthesized:
                self._on_speech_synthesized(synth_audio)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            self._update_status(latency_ms=latency_ms)
            logger.info(f"Processing latency: {latency_ms:.0f}ms")

            # Output to virtual microphone
            self._audio_output.play(synth_audio, sample_rate)

        except Exception as e:
            logger.error(f"Speech processing error: {e}")

        finally:
            self._update_status(is_processing=False)

    def get_status(self) -> PipelineStatus:
        """Get current pipeline status."""
        return self._status

    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._state == PipelineState.RUNNING

    def set_voice(self, voice: str) -> bool:
        """
        Change the TTS voice.

        Args:
            voice: Voice model name

        Returns:
            True if successful
        """
        if self._tts is None:
            return False

        return self._tts.set_voice(voice)

    def set_speed(self, speed: float) -> None:
        """Set TTS speech rate."""
        if self._tts is not None:
            self._tts.set_speed(speed)

    def set_input_device(self, device: Optional[str]) -> None:
        """Change input device."""
        was_running = self._state == PipelineState.RUNNING
        if was_running:
            self.stop()

        self.config.input_device = device
        if self._audio_capture is not None:
            self._audio_capture.device = device

        if was_running:
            self.start()

    def set_output_device(self, device: Optional[str]) -> None:
        """Change output device."""
        if self._audio_output is not None:
            self._audio_output.set_device(device)

    @staticmethod
    def list_input_devices():
        """List available input devices."""
        return AudioCapture.list_devices()

    @staticmethod
    def list_output_devices():
        """List available output devices."""
        return AudioOutput.list_devices()

    @staticmethod
    def list_voices():
        """List available TTS voices."""
        return PiperTTS.list_voices()

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
