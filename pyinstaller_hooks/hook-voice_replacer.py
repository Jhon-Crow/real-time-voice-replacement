"""PyInstaller hook for voice_replacer package.

This hook ensures all voice_replacer submodules are properly collected and included
in the PyInstaller bundle, preventing import errors at runtime.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all submodules of voice_replacer to ensure they're included in the bundle
hiddenimports = collect_submodules('voice_replacer')

# Collect any data files if present
datas = collect_data_files('voice_replacer')
