#!/usr/bin/env python3
"""
Simple TTS Demo

Demonstrates text-to-speech synthesis using Piper TTS.
This is useful for testing the TTS component in isolation.

Usage:
    python examples/simple_tts_demo.py "Hello, world!"
    python examples/simple_tts_demo.py --voice en_US-amy-medium "Testing female voice"
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from voice_replacer.tts import PiperTTS


def main():
    parser = argparse.ArgumentParser(description='Simple TTS Demo')
    parser.add_argument('text', nargs='?', default='Hello! This is a test of the text to speech system.',
                       help='Text to synthesize')
    parser.add_argument('--voice', default='en_US-lessac-medium',
                       help='Voice model to use')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech rate multiplier')
    parser.add_argument('--output', '-o', default='output.wav',
                       help='Output WAV file')
    parser.add_argument('--play', action='store_true',
                       help='Play audio after synthesis')

    args = parser.parse_args()

    print(f"Initializing TTS with voice: {args.voice}")
    tts = PiperTTS(voice=args.voice, speed=args.speed)

    print("Downloading/loading model...")
    if not tts.initialize():
        print("Failed to initialize TTS")
        return 1

    print(f"Synthesizing: '{args.text}'")
    audio, sample_rate = tts.synthesize(args.text)

    if len(audio) == 0:
        print("No audio generated")
        return 1

    print(f"Generated {len(audio) / sample_rate:.2f} seconds of audio at {sample_rate}Hz")

    # Save to file
    import wave
    import numpy as np

    with wave.open(args.output, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        audio_int16 = (audio * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"Saved to: {args.output}")

    # Play if requested
    if args.play:
        try:
            import sounddevice as sd
            print("Playing audio...")
            sd.play(audio, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Could not play audio: {e}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
