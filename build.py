#!/usr/bin/env python3
"""
Build script for creating standalone executable.

Creates a single .exe file using PyInstaller that bundles:
- Python runtime
- All dependencies
- Voice models (optional, can be downloaded on first run)

Usage:
    python build.py              # Build exe and create installer
    python build.py --clean      # Clean build directories
    python build.py --no-installer  # Skip installer creation

Works on both local development and GitHub Actions CI.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Try to read version from pyproject.toml
def get_version():
    """Get version from pyproject.toml."""
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    if pyproject_path.exists():
        try:
            import tomllib
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                return data.get("project", {}).get("version", "0.1.0")
        except (ImportError, KeyError):
            pass
        # Fallback for Python < 3.11
        try:
            import tomli
            with open(pyproject_path, "rb") as f:
                data = tomli.load(f)
                return data.get("project", {}).get("version", "0.1.0")
        except (ImportError, KeyError):
            pass
    return "0.1.0"

# Build configuration
APP_NAME = "VoiceReplacer"
APP_VERSION = get_version()
ENTRY_POINT = "src/voice_replacer/__main__.py"
ICON_PATH = "assets/icon.ico"  # Optional icon

# Paths
ROOT_DIR = Path(__file__).parent
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"


def clean():
    """Clean build directories."""
    print("Cleaning build directories...")
    for dir_path in [DIST_DIR, BUILD_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    print("Done")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    try:
        import PyInstaller
        print(f"  PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("  PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"],
                      check=True)

    # Check other dependencies
    deps = ["numpy", "sounddevice", "vosk", "PyQt6"]
    for dep in deps:
        try:
            __import__(dep.lower().replace("-", "_"))
            print(f"  {dep}: OK")
        except ImportError:
            print(f"  {dep}: Missing")

    print("Done")


def build_exe():
    """Build the executable."""
    print("Building executable...")

    # PyInstaller options
    options = [
        "pyinstaller",
        "--onefile",  # Single file executable
        "--windowed",  # No console window
        f"--name={APP_NAME}",
        "--clean",
        # Hidden imports that PyInstaller might miss
        "--hidden-import=vosk",
        "--hidden-import=sounddevice",
        "--hidden-import=numpy",
        "--hidden-import=scipy",
        "--hidden-import=scipy.signal",
        "--hidden-import=onnxruntime",
        "--hidden-import=PyQt6.QtWidgets",
        "--hidden-import=PyQt6.QtCore",
        "--hidden-import=PyQt6.QtGui",
        # Collect data files
        "--collect-data=vosk",
        "--collect-data=piper_phonemize",
        # Exclude unnecessary modules to reduce size
        "--exclude-module=matplotlib",
        "--exclude-module=tkinter",
        "--exclude-module=PIL",
        "--exclude-module=cv2",
    ]

    # Add icon if exists
    icon_path = ROOT_DIR / ICON_PATH
    if icon_path.exists():
        options.append(f"--icon={icon_path}")

    # Entry point
    options.append(str(ROOT_DIR / ENTRY_POINT))

    # Run PyInstaller
    result = subprocess.run(options, cwd=ROOT_DIR)

    if result.returncode == 0:
        exe_path = DIST_DIR / f"{APP_NAME}.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"\nBuild successful!")
            print(f"Executable: {exe_path}")
            print(f"Size: {size_mb:.1f} MB")
        else:
            print("\nBuild completed but executable not found")
            return 1
    else:
        print("\nBuild failed")
        return 1

    return 0


def create_installer():
    """Create an installer using Inno Setup or NSIS."""
    print("Creating installer...")

    # Check for Inno Setup first (preferred)
    iscc_path = shutil.which("iscc") or shutil.which("ISCC")
    if iscc_path:
        return create_inno_installer(iscc_path)

    # Fallback to NSIS
    nsis_path = shutil.which("makensis")
    if nsis_path:
        return create_nsis_installer(nsis_path)

    print("No installer tool found.")
    print("  - Install Inno Setup from https://jrsoftware.org/isinfo.php")
    print("  - Or install NSIS from https://nsis.sourceforge.io/")
    print("Skipping installer creation.")


def create_inno_installer(iscc_path):
    """Create installer using Inno Setup."""
    print(f"Using Inno Setup: {iscc_path}")

    iss_file = ROOT_DIR / "installer.iss"
    if not iss_file.exists():
        print(f"Inno Setup script not found: {iss_file}")
        return

    # Run Inno Setup compiler
    result = subprocess.run([iscc_path, str(iss_file)], cwd=ROOT_DIR)

    if result.returncode == 0:
        print(f"Installer created: dist/{APP_NAME}-{APP_VERSION}-setup.exe")
    else:
        print("Installer creation failed")


def create_nsis_installer(nsis_path):
    """Create installer using NSIS."""
    print(f"Using NSIS: {nsis_path}")

    # Create NSIS script
    nsis_script = f"""
!include "MUI2.nsh"

Name "{APP_NAME}"
OutFile "dist/{APP_NAME}-{APP_VERSION}-setup.exe"
InstallDir "$PROGRAMFILES\\{APP_NAME}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath $INSTDIR
    File "dist\\{APP_NAME}.exe"
    CreateShortcut "$DESKTOP\\{APP_NAME}.lnk" "$INSTDIR\\{APP_NAME}.exe"
    CreateShortcut "$SMPROGRAMS\\{APP_NAME}.lnk" "$INSTDIR\\{APP_NAME}.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\{APP_NAME}.exe"
    Delete "$DESKTOP\\{APP_NAME}.lnk"
    Delete "$SMPROGRAMS\\{APP_NAME}.lnk"
    RMDir "$INSTDIR"
SectionEnd
"""

    nsis_file = ROOT_DIR / "installer.nsi"
    with open(nsis_file, "w") as f:
        f.write(nsis_script)

    # Run NSIS
    result = subprocess.run(["makensis", str(nsis_file)], cwd=ROOT_DIR)

    # Clean up
    nsis_file.unlink()

    if result.returncode == 0:
        print(f"Installer created: dist/{APP_NAME}-{APP_VERSION}-setup.exe")
    else:
        print("Installer creation failed")


def main():
    """Main build process."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Voice Replacer executable")
    parser.add_argument("--clean", action="store_true", help="Clean build directories")
    parser.add_argument("--no-installer", action="store_true",
                       help="Skip installer creation")
    args = parser.parse_args()

    if args.clean:
        clean()
        return 0

    check_dependencies()

    result = build_exe()
    if result != 0:
        return result

    if not args.no_installer:
        create_installer()

    print("\nBuild complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
