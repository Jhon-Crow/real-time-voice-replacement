#!/usr/bin/env python3
"""
ASR (Automatic Speech Recognition) Demo

Demonstrates offline speech recognition using Vosk.
Records audio from microphone and transcribes it.

Usage:
    python examples/asr_demo.py
    python examples/asr_demo.py --file audio.wav
"""
import sys
import argparse
import wave
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from voice_replacer.asr import SpeechRecognizer


def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone."""
    import sounddevice as sd

    print(f"Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("Recording complete")

    return audio.flatten()


def load_audio(file_path: str) -> tuple:
    """Load audio from WAV file."""
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)

    # Convert to float32
    audio = audio.astype(np.float32) / 32767.0

    return audio, sample_rate


def main():
    parser = argparse.ArgumentParser(description='ASR Demo')
    parser.add_argument('--file', '-f', help='Audio file to transcribe')
    parser.add_argument('--duration', '-d', type=float, default=5.0,
                       help='Recording duration in seconds')
    parser.add_argument('--model', default='en-us-small',
                       help='Vosk model to use')

    args = parser.parse_args()

    print(f"Initializing ASR with model: {args.model}")
    asr = SpeechRecognizer(model_name=args.model)

    print("Downloading/loading model...")
    if not asr.initialize():
        print("Failed to initialize ASR")
        return 1

    if args.file:
        print(f"Loading audio from: {args.file}")
        audio, sample_rate = load_audio(args.file)

        # Resample if needed
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples).astype(np.float32)
    else:
        audio = record_audio(args.duration)

    print(f"Audio: {len(audio) / 16000:.2f} seconds")
    print("Transcribing...")

    text, confidence = asr.recognize(audio)

    print()
    print("=" * 40)
    print(f"Transcription: {text}")
    print(f"Confidence: {confidence:.2f}")
    print("=" * 40)

    return 0


if __name__ == '__main__':
    sys.exit(main())
