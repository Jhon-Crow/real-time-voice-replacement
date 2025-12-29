"""
Graphical User Interface for Voice Replacement System.

Provides a system tray icon and settings window.
"""
import logging
import sys
import threading
from typing import Optional, List
from functools import partial

logger = logging.getLogger(__name__)

# Try to import PyQt6, fall back to tkinter
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QSlider, QGroupBox,
        QSystemTrayIcon, QMenu, QMessageBox, QProgressDialog
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    logger.warning("PyQt6 not available, using tkinter fallback")

from .pipeline import VoiceReplacementPipeline, PipelineStatus, PipelineState
from .config import AppConfig, VOICE_PRESETS
from .audio_capture import AudioCapture
from .audio_output import AudioOutput
from .tts import PiperTTS


# Only define PyQt6-dependent classes when PyQt6 is available
if HAS_PYQT:
    class StatusSignal(QObject):
        """Signal for status updates from pipeline thread."""
        status_changed = pyqtSignal(object)
        text_recognized = pyqtSignal(str)


    class VoiceReplacerGUI(QMainWindow):
        """Main application window."""

        def __init__(self, config: Optional[AppConfig] = None):
            """
            Initialize GUI.

            Args:
                config: Application configuration
            """
            super().__init__()

            self.config = config or AppConfig.load()
            self.pipeline = VoiceReplacementPipeline(self.config)

            # Signals for thread-safe updates
            self._signals = StatusSignal()
            self._signals.status_changed.connect(self._on_status_update)
            self._signals.text_recognized.connect(self._on_text_update)

            # Set up pipeline callbacks
            self.pipeline.set_status_callback(
                lambda s: self._signals.status_changed.emit(s)
            )
            self.pipeline.set_text_callback(
                lambda t: self._signals.text_recognized.emit(t)
            )

            self._setup_ui()
            self._setup_tray()
            self._load_devices()

            # Initialize pipeline in background
            self._init_thread = threading.Thread(target=self._initialize_pipeline)
            self._init_thread.start()

        def _setup_ui(self):
            """Set up the main UI."""
            self.setWindowTitle("Voice Replacer")
            self.setMinimumSize(400, 500)

            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            # Status section
            status_group = QGroupBox("Status")
            status_layout = QVBoxLayout(status_group)

            # Enable/disable button
            self.enable_btn = QPushButton("Enable Voice Replacement")
            self.enable_btn.setCheckable(True)
            self.enable_btn.setEnabled(False)  # Disabled until initialized
            self.enable_btn.clicked.connect(self._toggle_enabled)
            status_layout.addWidget(self.enable_btn)

            # Status indicators
            indicator_layout = QHBoxLayout()

            self.speaking_indicator = QLabel("🔇 Idle")
            self.processing_indicator = QLabel("")
            indicator_layout.addWidget(self.speaking_indicator)
            indicator_layout.addWidget(self.processing_indicator)
            indicator_layout.addStretch()

            status_layout.addLayout(indicator_layout)

            # Last recognized text
            self.text_label = QLabel("Last text: (none)")
            self.text_label.setWordWrap(True)
            status_layout.addWidget(self.text_label)

            # Latency
            self.latency_label = QLabel("Latency: --")
            status_layout.addWidget(self.latency_label)

            layout.addWidget(status_group)

            # Input section
            input_group = QGroupBox("Input")
            input_layout = QVBoxLayout(input_group)

            input_layout.addWidget(QLabel("Microphone:"))
            self.input_combo = QComboBox()
            self.input_combo.currentIndexChanged.connect(self._on_input_changed)
            input_layout.addWidget(self.input_combo)

            layout.addWidget(input_group)

            # Output section
            output_group = QGroupBox("Output")
            output_layout = QVBoxLayout(output_group)

            output_layout.addWidget(QLabel("Virtual Microphone:"))
            self.output_combo = QComboBox()
            self.output_combo.currentIndexChanged.connect(self._on_output_changed)
            output_layout.addWidget(self.output_combo)

            # VB-Audio info
            vb_info = QLabel(
                "💡 Install VB-Audio Virtual Cable for best results.\n"
                "Other apps will use this as their microphone."
            )
            vb_info.setStyleSheet("color: gray; font-size: 10px;")
            vb_info.setWordWrap(True)
            output_layout.addWidget(vb_info)

            layout.addWidget(output_group)

            # Voice section
            voice_group = QGroupBox("Voice")
            voice_layout = QVBoxLayout(voice_group)

            voice_layout.addWidget(QLabel("Voice Model:"))
            self.voice_combo = QComboBox()
            self._load_voices()
            self.voice_combo.currentIndexChanged.connect(self._on_voice_changed)
            voice_layout.addWidget(self.voice_combo)

            # Speed slider
            speed_layout = QHBoxLayout()
            speed_layout.addWidget(QLabel("Speed:"))
            self.speed_slider = QSlider(Qt.Orientation.Horizontal)
            self.speed_slider.setMinimum(50)
            self.speed_slider.setMaximum(200)
            self.speed_slider.setValue(100)
            self.speed_slider.valueChanged.connect(self._on_speed_changed)
            speed_layout.addWidget(self.speed_slider)
            self.speed_label = QLabel("1.0x")
            speed_layout.addWidget(self.speed_label)
            voice_layout.addLayout(speed_layout)

            layout.addWidget(voice_group)

            # Spacer
            layout.addStretch()

            # Info
            info_label = QLabel(
                "Voice Replacer v0.1.0\n"
                "Open Source - MIT License"
            )
            info_label.setStyleSheet("color: gray; font-size: 10px;")
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(info_label)

        def _setup_tray(self):
            """Set up system tray icon."""
            self.tray_icon = QSystemTrayIcon(self)

            # Create a simple icon
            pixmap = QPixmap(32, 32)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            painter.setBrush(QColor(100, 150, 255))
            painter.setPen(QColor(50, 100, 200))
            painter.drawEllipse(4, 4, 24, 24)
            painter.end()

            self.tray_icon.setIcon(QIcon(pixmap))
            self.tray_icon.setToolTip("Voice Replacer")

            # Tray menu
            tray_menu = QMenu()

            self.tray_enable_action = QAction("Enable", self)
            self.tray_enable_action.setCheckable(True)
            self.tray_enable_action.triggered.connect(self._toggle_enabled)
            tray_menu.addAction(self.tray_enable_action)

            tray_menu.addSeparator()

            show_action = QAction("Show Window", self)
            show_action.triggered.connect(self.show)
            tray_menu.addAction(show_action)

            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self._quit)
            tray_menu.addAction(quit_action)

            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.activated.connect(self._on_tray_activated)
            self.tray_icon.show()

        def _load_devices(self):
            """Load audio devices into combo boxes."""
            # Input devices
            self.input_combo.clear()
            self.input_combo.addItem("Default", None)

            for device in AudioCapture.list_devices():
                self.input_combo.addItem(device['name'], device['index'])

            # Output devices
            self.output_combo.clear()
            self.output_combo.addItem("Default", None)

            virtual_cable = AudioOutput.find_virtual_cable()
            for device in AudioOutput.list_devices():
                name = device['name']
                if virtual_cable and device['index'] == virtual_cable['index']:
                    name = f"⭐ {name} (Virtual Cable)"
                self.output_combo.addItem(name, device['index'])

                # Auto-select virtual cable
                if virtual_cable and device['index'] == virtual_cable['index']:
                    self.output_combo.setCurrentIndex(self.output_combo.count() - 1)

        def _load_voices(self):
            """Load available voices."""
            self.voice_combo.clear()

            for voice_id, info in PiperTTS.list_voices().items():
                self.voice_combo.addItem(info['description'], voice_id)

        def _initialize_pipeline(self):
            """Initialize pipeline in background."""
            def progress(name, value):
                logger.info(f"Initializing {name}: {value * 100:.0f}%")

            success = self.pipeline.initialize(progress)

            if success:
                # Enable UI on main thread
                QTimer.singleShot(0, self._on_pipeline_ready)
            else:
                QTimer.singleShot(0, self._on_pipeline_error)

        def _on_pipeline_ready(self):
            """Called when pipeline is ready."""
            self.enable_btn.setEnabled(True)
            self.enable_btn.setText("Enable Voice Replacement")
            logger.info("Pipeline ready")

        def _on_pipeline_error(self):
            """Called when pipeline initialization fails."""
            self.enable_btn.setText("Initialization Failed")
            QMessageBox.critical(
                self,
                "Error",
                "Failed to initialize voice replacement system.\n"
                "Check the console for details."
            )

        def _toggle_enabled(self):
            """Toggle voice replacement on/off."""
            if self.pipeline.is_running():
                self.pipeline.stop()
                self.enable_btn.setChecked(False)
                self.enable_btn.setText("Enable Voice Replacement")
                self.tray_enable_action.setChecked(False)
            else:
                if self.pipeline.start():
                    self.enable_btn.setChecked(True)
                    self.enable_btn.setText("Disable Voice Replacement")
                    self.tray_enable_action.setChecked(True)

        def _on_status_update(self, status: PipelineStatus):
            """Handle status update from pipeline."""
            if status.is_speaking:
                self.speaking_indicator.setText("🎤 Speaking...")
            else:
                self.speaking_indicator.setText("🔇 Idle")

            if status.is_processing:
                self.processing_indicator.setText("⚙️ Processing")
            else:
                self.processing_indicator.setText("")

            if status.latency_ms > 0:
                self.latency_label.setText(f"Latency: {status.latency_ms:.0f}ms")

        def _on_text_update(self, text: str):
            """Handle recognized text."""
            self.text_label.setText(f"Last text: {text}")

        def _on_input_changed(self, index: int):
            """Handle input device change."""
            device = self.input_combo.itemData(index)
            self.pipeline.set_input_device(device)

        def _on_output_changed(self, index: int):
            """Handle output device change."""
            device = self.output_combo.itemData(index)
            self.pipeline.set_output_device(device)

        def _on_voice_changed(self, index: int):
            """Handle voice change."""
            voice = self.voice_combo.itemData(index)
            if voice:
                self.pipeline.set_voice(voice)

        def _on_speed_changed(self, value: int):
            """Handle speed change."""
            speed = value / 100.0
            self.speed_label.setText(f"{speed:.1f}x")
            self.pipeline.set_speed(speed)

        def _on_tray_activated(self, reason):
            """Handle tray icon activation."""
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
                self.show()
                self.activateWindow()

        def closeEvent(self, event):
            """Handle window close."""
            if self.config.minimize_to_tray:
                event.ignore()
                self.hide()
                self.tray_icon.showMessage(
                    "Voice Replacer",
                    "Running in background. Right-click tray icon for options.",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000
                )
            else:
                self._quit()

        def _quit(self):
            """Quit the application."""
            self.pipeline.stop()
            self.config.save()
            self.tray_icon.hide()
            QApplication.quit()


def run_gui(config: Optional[AppConfig] = None):
    """
    Run the GUI application.

    Args:
        config: Optional configuration
    """
    if not HAS_PYQT:
        logger.error("PyQt6 is required for GUI. Install with: pip install PyQt6")
        return run_cli(config)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    window = VoiceReplacerGUI(config)

    if config and config.start_minimized:
        window.hide()
    else:
        window.show()

    return app.exec()


def run_cli(config: Optional[AppConfig] = None):
    """
    Run in CLI mode (fallback when no GUI available).

    Args:
        config: Optional configuration
    """
    print("Voice Replacer - CLI Mode")
    print("=" * 40)
    print()

    config = config or AppConfig.load()
    pipeline = VoiceReplacementPipeline(config)

    print("Initializing...")
    if not pipeline.initialize():
        print("Failed to initialize pipeline")
        return 1

    print("Starting voice replacement...")
    print("Press Ctrl+C to stop")
    print()

    def on_text(text):
        print(f"Recognized: {text}")

    pipeline.set_text_callback(on_text)

    if not pipeline.start():
        print("Failed to start pipeline")
        return 1

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")

    pipeline.stop()
    print("Done")
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(run_gui())
