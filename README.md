# Real-Time Voice Replacement

A standalone Windows application that replaces your voice from the microphone with a neural network-synthesized voice in real-time. Works with Discord, Zoom, games, and any other application that uses microphone input.

## Features

- **Complete Voice Replacement**: Your voice is replaced with an AI-generated voice, not just modified
- **Fully Local Processing**: All processing happens on your PC - no internet required, no API keys, no registration
- **Low Latency**: Optimized for real-time use with latency < 500ms
- **Multiple Voice Options**: Choose from several pre-trained voice models
- **Virtual Microphone Output**: Works with VB-Audio Virtual Cable so any app can use the synthesized voice
- **System Tray Integration**: Runs in background with easy access
- **Lightweight**: Minimal resource usage for background operation

## How It Works

```
Microphone → Voice Activity Detection → Speech Recognition → Text-to-Speech → Virtual Microphone
```

1. **Voice Activity Detection (VAD)**: Silero VAD detects when you're speaking
2. **Speech Recognition (ASR)**: Vosk converts your speech to text offline
3. **Text-to-Speech (TTS)**: Piper TTS generates new speech with a different voice
4. **Audio Output**: The synthesized audio is sent to a virtual microphone

## Requirements

- **OS**: Windows 10/11 (x64)
- **RAM**: Minimum 4GB, recommended 8GB+
- **CPU**: Any modern CPU (Intel i5/Ryzen 5 or better recommended)
- **Virtual Audio Device**: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free)

## Installation

### Option 1: Standalone Executable (Recommended)

1. Download the latest release from [Releases](https://github.com/Jhon-Crow/real-time-voice-replacement/releases)
2. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
3. Run `VoiceReplacer.exe`

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/Jhon-Crow/real-time-voice-replacement.git
cd real-time-voice-replacement

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[all]"

# Run the application
python -m voice_replacer
```

## Usage

### First Run

1. On first launch, the app will download required models (~100MB total):
   - Speech recognition model (Vosk)
   - Text-to-speech voice model (Piper)

2. Install VB-Audio Virtual Cable if not already installed

3. In other applications (Discord, Zoom, etc.), select "VB-Audio Virtual Cable" as your microphone

### GUI Mode

```bash
python -m voice_replacer
```

- **Enable/Disable**: Toggle voice replacement on/off
- **Microphone**: Select your input microphone
- **Virtual Microphone**: Select the virtual audio device for output
- **Voice**: Choose the synthesized voice
- **Speed**: Adjust speech rate

### CLI Mode

```bash
# Run with GUI (default)
python -m voice_replacer

# Run in CLI mode (no GUI)
python -m voice_replacer --cli

# List available devices
python -m voice_replacer --list-devices

# List available voices
python -m voice_replacer --list-voices

# Use specific voice
python -m voice_replacer --voice en_US-lessac-medium

# Enable debug logging
python -m voice_replacer --debug
```

## Configuration

Configuration is stored in:
- Windows: `%LOCALAPPDATA%\VoiceReplacer\config.json`
- Linux/Mac: `~/.local/share/VoiceReplacer/config.json`

### Available Voices

| Voice ID | Description |
|----------|-------------|
| `en_US-lessac-medium` | US English male voice (default) |
| `en_US-amy-medium` | US English female voice |
| `en_GB-alan-medium` | British English male voice |
| `en_US-libritts-high` | US English multi-speaker (high quality) |

## Building from Source

### Build Standalone Executable

```bash
# Install build dependencies
pip install pyinstaller

# Build executable
python build.py
```

The executable will be created in the `dist/` folder.

## Architecture

```
voice_replacer/
├── __init__.py         # Package initialization
├── __main__.py         # Entry point
├── config.py           # Configuration management
├── audio_capture.py    # Microphone input handling
├── audio_output.py     # Virtual microphone output
├── vad.py              # Voice Activity Detection (Silero)
├── asr.py              # Speech Recognition (Vosk)
├── tts.py              # Text-to-Speech (Piper)
├── pipeline.py         # Real-time processing pipeline
└── gui.py              # PyQt6 GUI with system tray
```

## Performance Targets

- RAM: < 500 MB
- CPU: < 15% on idle, < 30% during processing
- Latency: < 500ms end-to-end

## Troubleshooting

### No audio output
- Ensure VB-Audio Virtual Cable is installed
- Check that the correct output device is selected in Voice Replacer
- Verify the target application is using "VB-Audio Virtual Cable" as microphone

### High latency
- Try using a smaller voice model
- Increase speed setting to reduce synthesis time
- Ensure no other heavy applications are running

### Voice quality issues
- Try different voice models
- Speak clearly and at normal pace
- Check microphone input levels

## Dependencies

All processing is local using these open-source libraries:

- **[Vosk](https://alphacephei.com/vosk/)**: Offline speech recognition
- **[Piper](https://github.com/rhasspy/piper)**: Fast local neural TTS
- **[Silero VAD](https://github.com/snakers4/silero-vad)**: Voice activity detection
- **[sounddevice](https://python-sounddevice.readthedocs.io/)**: Audio I/O
- **[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)**: GUI framework

## License

This project is released into the public domain under [The Unlicense](LICENSE).

Third-party libraries and models have their own licenses:
- Vosk: Apache 2.0
- Piper: MIT
- Silero VAD: MIT
- PyQt6: GPL v3

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Roadmap

- [ ] Support for more languages
- [ ] Voice cloning from audio samples
- [ ] Prosody/intonation preservation
- [ ] GPU acceleration (optional)
- [ ] More voice presets
- [ ] Real-time pitch/speed adjustment
