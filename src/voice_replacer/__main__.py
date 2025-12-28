"""
Main entry point for Voice Replacer application.
"""
import argparse
import logging
import sys

from .config import AppConfig
from .gui import run_gui, run_cli


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time Voice Replacement System"
    )
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run in CLI mode (no GUI)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--voice',
        type=str,
        default=None,
        help='Voice model to use'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices and exit'
    )
    parser.add_argument(
        '--list-voices',
        action='store_true',
        help='List available voice models and exit'
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # List devices if requested
    if args.list_devices:
        from .audio_capture import AudioCapture
        from .audio_output import AudioOutput

        print("\nInput Devices (Microphones):")
        print("-" * 40)
        for device in AudioCapture.list_devices():
            default = " (default)" if device.get('is_default') else ""
            print(f"  [{device['index']}] {device['name']}{default}")

        print("\nOutput Devices:")
        print("-" * 40)
        for device in AudioOutput.list_devices():
            default = " (default)" if device.get('is_default') else ""
            print(f"  [{device['index']}] {device['name']}{default}")

        virtual = AudioOutput.find_virtual_cable()
        if virtual:
            print(f"\n✓ Virtual Cable found: {virtual['name']}")
        else:
            print("\n⚠ No virtual cable found. Install VB-Audio Virtual Cable.")

        return 0

    # List voices if requested
    if args.list_voices:
        from .tts import PiperTTS

        print("\nAvailable Voice Models:")
        print("-" * 40)
        for voice_id, info in PiperTTS.list_voices().items():
            print(f"  {voice_id}")
            print(f"    {info['description']}")
            print()
        return 0

    # Load or create configuration
    config = AppConfig.load()

    if args.voice:
        config.tts.model_name = args.voice

    if args.debug:
        config.debug_mode = True
        config.log_level = 'DEBUG'

    # Run application
    if args.cli:
        return run_cli(config)
    else:
        return run_gui(config)


if __name__ == '__main__':
    sys.exit(main())
