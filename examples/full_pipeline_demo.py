#!/usr/bin/env python3
"""
Full Pipeline Demo

Demonstrates the complete voice replacement pipeline:
Microphone -> VAD -> ASR -> TTS -> Output

Usage:
    python examples/full_pipeline_demo.py
    python examples/full_pipeline_demo.py --cli
"""
import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from voice_replacer.pipeline import VoiceReplacementPipeline, PipelineStatus
from voice_replacer.config import AppConfig


def on_status_change(status: PipelineStatus):
    """Handle status changes."""
    if status.is_speaking:
        print("🎤 Speaking...", end='\r')
    elif status.is_processing:
        print("⚙️  Processing...", end='\r')
    else:
        print("🔇 Idle       ", end='\r')


def on_text_recognized(text: str):
    """Handle recognized text."""
    print(f"\n📝 Recognized: {text}")


def main():
    parser = argparse.ArgumentParser(description='Full Pipeline Demo')
    parser.add_argument('--voice', default='en_US-lessac-medium',
                       help='Voice model to use')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech rate multiplier')
    parser.add_argument('--list-devices', action='store_true',
                       help='List audio devices and exit')

    args = parser.parse_args()

    if args.list_devices:
        from voice_replacer.audio_capture import AudioCapture
        from voice_replacer.audio_output import AudioOutput

        print("\nInput Devices:")
        for device in AudioCapture.list_devices():
            print(f"  [{device['index']}] {device['name']}")

        print("\nOutput Devices:")
        for device in AudioOutput.list_devices():
            print(f"  [{device['index']}] {device['name']}")

        virtual = AudioOutput.find_virtual_cable()
        if virtual:
            print(f"\n✓ Virtual Cable found: {virtual['name']}")
        else:
            print("\n⚠ No virtual cable found")

        return 0

    # Create configuration
    config = AppConfig()
    config.tts.model_name = args.voice
    config.tts.speed = args.speed

    # Create pipeline
    print("Creating pipeline...")
    pipeline = VoiceReplacementPipeline(config)

    # Set callbacks
    pipeline.set_status_callback(on_status_change)
    pipeline.set_text_callback(on_text_recognized)

    # Initialize
    print("Initializing...")

    def progress(name, value):
        print(f"  {name}: {value * 100:.0f}%")

    if not pipeline.initialize(progress):
        print("Failed to initialize pipeline")
        return 1

    print("\nPipeline ready!")
    print("=" * 50)
    print("Speak into your microphone. Your voice will be")
    print("recognized and synthesized with a new voice.")
    print()
    print("The synthesized audio will be sent to your default")
    print("output device (or VB-Audio Virtual Cable if found).")
    print()
    print("Press Ctrl+C to stop.")
    print("=" * 50)
    print()

    # Start pipeline
    if not pipeline.start():
        print("Failed to start pipeline")
        return 1

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")

    pipeline.stop()
    print("Done")

    # Show stats
    status = pipeline.get_status()
    if status.latency_ms > 0:
        print(f"\nLast latency: {status.latency_ms:.0f}ms")
    if status.last_text:
        print(f"Last text: {status.last_text}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
