"""
Main entry point for Voice Replacer application.
"""
import argparse
import logging
import sys
import os
import traceback
from pathlib import Path


def _get_log_dir():
    """Get the directory for log files."""
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')))
    else:  # Linux/Mac
        base = Path(os.path.expanduser('~/.local/share'))
    log_dir = base / 'VoiceReplacer' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _setup_crash_logging():
    """Set up early crash logging before any imports that might fail.

    This ensures that if the application crashes during startup (e.g., due to
    missing dependencies or DLL issues in PyInstaller builds), we can still
    capture and log the error for diagnosis.
    """
    log_dir = _get_log_dir()
    crash_log = log_dir / 'crash.log'

    # Set up basic file logging for crashes
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(crash_log, mode='a', encoding='utf-8'),
        ]
    )
    return crash_log


def _show_error_dialog(title, message, crash_log_path=None):
    """Show an error dialog to the user.

    Works in windowed mode (no console) by using tkinter as a fallback
    if PyQt6 is not available.
    """
    details = message
    if crash_log_path:
        details += f"\n\nCrash log saved to:\n{crash_log_path}"

    # Try PyQt6 first
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message[:500] if len(message) > 500 else message)
        if len(message) > 500 or crash_log_path:
            msg_box.setDetailedText(details)
        msg_box.exec()
        return
    except Exception:
        pass

    # Fall back to tkinter
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror(title, details[:1000])
        root.destroy()
        return
    except Exception:
        pass

    # Last resort: write to stderr (may not be visible in windowed mode)
    print(f"{title}: {details}", file=sys.stderr)


def _setup_package_path():
    """Set up the Python path for PyInstaller compatibility.

    When running as a PyInstaller-bundled executable, the package context
    may not be properly established. This function ensures the src directory
    is in sys.path so that absolute imports like 'from voice_replacer.xxx' work correctly.

    For PyInstaller, sys._MEIPASS contains the path to the extracted bundle.
    For normal execution, we add the parent of the voice_replacer package to the path.
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Running from source
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if base_path not in sys.path:
        sys.path.insert(0, base_path)


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

    # Set up logging (reconfigure with user preferences)
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_dir = _get_log_dir()

    # Clear existing handlers and reconfigure
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log', mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Voice Replacer starting...")

    # Import voice_replacer modules after path setup
    from voice_replacer.config import AppConfig
    from voice_replacer.gui import run_gui, run_cli

    # List devices if requested
    if args.list_devices:
        from voice_replacer.audio_capture import AudioCapture
        from voice_replacer.audio_output import AudioOutput

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
        from voice_replacer.tts import PiperTTS

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


def run():
    """
    Entry point wrapper with global exception handling.

    This function wraps main() to catch any unhandled exceptions and:
    1. Log them to a crash log file
    2. Show an error dialog to the user (important for windowed mode where
       there's no console to see error messages)
    """
    # Set up crash logging early, before any imports that might fail
    crash_log_path = _setup_crash_logging()
    logger = logging.getLogger(__name__)

    try:
        # Set up package path before any voice_replacer imports
        _setup_package_path()

        return main()

    except Exception as e:
        # Log the full traceback
        error_msg = traceback.format_exc()
        logger.critical(f"Unhandled exception: {error_msg}")

        # Show error dialog to user (works in windowed mode)
        _show_error_dialog(
            "Voice Replacer - Startup Error",
            f"The application encountered an error and could not start.\n\n"
            f"Error: {type(e).__name__}: {str(e)}",
            crash_log_path
        )

        return 1


if __name__ == '__main__':
    sys.exit(run())
