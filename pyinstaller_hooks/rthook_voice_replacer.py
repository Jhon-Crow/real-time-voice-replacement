"""PyInstaller runtime hook for voice_replacer.

This runtime hook runs at application startup before any imports,
ensuring the Python path is correctly set up for the bundled environment.
"""

import sys
import os


def setup_voice_replacer_path():
    """Ensure voice_replacer package can be imported in PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        # sys._MEIPASS is the path where PyInstaller extracts bundled files
        base_path = sys._MEIPASS

        # Ensure base_path is in sys.path at the beginning
        if base_path not in sys.path:
            sys.path.insert(0, base_path)

        # Also add the voice_replacer package path explicitly
        voice_replacer_path = os.path.join(base_path, 'voice_replacer')
        if os.path.exists(voice_replacer_path) and voice_replacer_path not in sys.path:
            sys.path.insert(0, voice_replacer_path)


# Execute the path setup
setup_voice_replacer_path()
