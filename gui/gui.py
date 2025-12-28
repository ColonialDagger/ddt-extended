import threading
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QFormLayout, QComboBox, QSpinBox,
    QPushButton, QSizePolicy, QCheckBox
)

from gui.graphs import HealthGraphCanvas, DpsGraphCanvas
from gui.overlay import OverlayWindow
import scanner

class MainWindow(QMainWindow):
    """
    Main application window for DDT Extended.
    """

    # Thread‑safe status update signal
    status_update = Signal(str)

    def __init__(self):
        """
        Initialize the main window and UI components.
        """
        super().__init__()

        self._scanner = None
        self._scanner_ready = False

        self.setWindowTitle("DDT Extended (Modern Qt Layout)")
        self.resize(1100, 600)

        # Simple history for health graph
        self._health_times = []
        self._health_values = []

        self._dps_times = []
        self._dps_values = []

        # Delay overlay creation until UI is built
        self.overlay = None
        QTimer.singleShot(0, self._create_overlay)

        # MAIN UI LAYOUT
        central = QWidget()
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # LEFT PANEL
        left_panel = QVBoxLayout()

        stats_group = QGroupBox("Live Stats")
        stats_layout = QVBoxLayout()

        self.health_label = QLabel("Health: --")
        self.fps_label = QLabel("FPS: --")
        self.delta_label = QLabel("Δt: --")

        # Status label
        self.status_label = QLabel("Status: Searching for Destiny 2…")

        stats_layout.addWidget(self.health_label)
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.delta_label)
        stats_layout.addWidget(self.status_label)
        stats_group.setLayout(stats_layout)

        graphs_group = QGroupBox("Graphs")
        graphs_layout = QVBoxLayout()

        self.health_canvas = HealthGraphCanvas()
        self.health_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.dps_canvas = DpsGraphCanvas()
        self.dps_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        graphs_layout.addWidget(self.health_canvas, 1)
        graphs_layout.addWidget(self.dps_canvas, 1)
        graphs_group.setLayout(graphs_layout)

        left_panel.addWidget(stats_group)
        left_panel.addWidget(graphs_group, 3)
        left_panel.addStretch()

        # RIGHT PANEL
        right_panel = QVBoxLayout()
        settings_group = QGroupBox("Settings")

        form = QFormLayout()

        self.brightness_box = QComboBox()
        self.brightness_box.addItems([str(i) for i in range(1, 8)])
        form.addRow("Brightness:", self.brightness_box)

        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(1, 120)
        self.buffer_spin.setValue(10)
        form.addRow("Health Buffer:", self.buffer_spin)

        self.overlay_toggle = QCheckBox("Enable Overlay")
        form.addRow("", self.overlay_toggle)

        self.apply_button = QPushButton("Apply Settings")
        form.addRow("", self.apply_button)

        settings_group.setLayout(form)

        right_panel.addWidget(settings_group)
        right_panel.addStretch()

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)

        # Timer for scanner polling
        self._t0 = time.time()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_from_scanner)
        self._timer.setInterval(50)
        self._timer.start()

        # Connect signal to label
        self.status_update.connect(self.status_label.setText)

        self.apply_button.clicked.connect(self.apply_settings)

        # Start scanner auto-retry thread
        brightness = int(self.brightness_box.currentText())
        buffer_size = int(self.buffer_spin.value())

        threading.Thread(
            target=self._scanner_retry_loop,
            args=(brightness, buffer_size),
            daemon=True
        ).start()

    def _create_overlay(self):
        """
        Create the overlay window.

        :return:
        """
        self.overlay = OverlayWindow()

    def _scanner_retry_loop(self, brightness, buffer_size):
        """
        Retry loop to create and start the scanner.

        :param brightness:
        :param buffer_size:
        :return:
        """

        while not self._scanner_ready:
            try:
                print("Trying to create scanner...")
                self.status_update.emit("Status: Searching for Destiny 2…")

                s = scanner.Scanner(
                    brightness=brightness,
                    health_buffer_size=buffer_size
                )

                print("Scanner created, starting capture...")
                s.start_capture()

                self._scanner = s
                self._scanner_ready = True

                self.status_update.emit("Status: Connected")
                print("Scanner is now running.")
                return

            except Exception as e:
                print("Scanner failed:", e)
                time.sleep(5)

    def _update_from_scanner(self):
        """
        Poll scanner for new data and update UI.

        :return:
        """

        if not self._scanner_ready or not self._scanner:
            self.health_label.setText("Health: --")
            self.fps_label.setText("FPS: --")
            self.delta_label.setText("Δt: --")
            return

        # Read scanner values
        health = self._scanner.get_health()
        fps = self._scanner.get_fps()
        delta = self._scanner.get_delta_t()

        # Update labels
        self.health_label.setText(f"Health: {health:.2f}%")
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.delta_label.setText(f"Δt: {delta * 1000:.2f} ms")

        # Time base
        now = time.time() - self._t0

        # Health Graph (%/t
        self._health_times.append(now)
        self._health_values.append(health)

        if len(self._health_times) > 2000:
            self._health_times = self._health_times[-2000:]
            self._health_values = self._health_values[-2000:]

        self.health_canvas.update_series(self._health_times, self._health_values)

        # Derivative graph (d%/dt)
        if len(self._health_values) >= 2:
            dh = self._health_values[-1] - self._health_values[-2]
            dt = self._health_times[-1] - self._health_times[-2]
            raw_deriv = dh / dt if dt > 0 else 0

            # Clamp positive values (boss cannot heal)
            deriv = min(raw_deriv, 0)
        else:
            deriv = 0

        self._dps_times.append(now)
        self._dps_values.append(deriv)

        if len(self._dps_times) > 2000:
            self._dps_times = self._dps_times[-2000:]
            self._dps_values = self._dps_values[-2000:]

        self.dps_canvas.update_series(self._dps_times, self._dps_values)

        # OVERLAY
        if self.overlay:
            self.overlay.set_text([
                f"Health: {health:.2f}%",
                f"FPS: {fps:.1f}",
                f"Δt: {delta * 1000:.2f} ms",
            ])

    def apply_settings(self):
        """
        Apply new settings from the UI.

        :return:
        """

        # Overlay toggle
        if self.overlay:
            if self.overlay_toggle.isChecked():
                self.overlay.show_overlay()
            else:
                self.overlay.hide_overlay()

        print("Applying settings — restarting scanner...")
        self.status_update.emit("Status: Restarting scanner…")

        # Stop existing scanner
        if self._scanner:
            try:
                self._scanner.stop_capture()
            except Exception as e:
                print("Error stopping scanner:", e)

        # Reset scanner state
        self._scanner = None
        self._scanner_ready = False

        # Clear graph history
        self._health_times.clear()
        self._health_values.clear()

        # Read new settings
        brightness = int(self.brightness_box.currentText())
        buffer_size = int(self.buffer_spin.value())

        # Start retry loop again
        threading.Thread(
            target=self._scanner_retry_loop,
            args=(brightness, buffer_size),
            daemon=True
        ).start()

    def closeEvent(self, event):
        """
        Cleanup on window close.

        :param event:
        :return:
        """
        try:
            if self._scanner:
                self._scanner.stop_capture()
        except:
            pass
        super().closeEvent(event)
