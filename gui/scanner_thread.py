# gui/scanner_thread.py
from PySide6.QtCore import QObject, Signal
import threading
import time
import ctypes
import scanner

class ScannerThread(QObject):
    health_changed = Signal(float)
    fps_changed = Signal(float)
    delta_changed = Signal(float)

    def __init__(self, brightness=4, buffer_size=30):
        super().__init__()
        self._brightness = brightness
        self._buffer_size = buffer_size
        self._running = False
        self._thread = None
        self._scanner = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def _thread_main(self):
        # WinRT requires STA apartment
        ctypes.windll.ole32.CoInitializeEx(None, 2)

        # Create scanner in this thread
        self._scanner = scanner.Scanner(self._brightness, self._buffer_size)

        while self._running:
            health = self._scanner.get_health()
            fps = self._scanner.get_fps()
            delta = self._scanner.get_delta_t()

            if health is not None:
                self.health_changed.emit(health)
            if fps is not None:
                self.fps_changed.emit(fps)
            if delta is not None:
                self.delta_changed.emit(delta)

            time.sleep(0.05)

    def stop(self):
        self._running = False
        if self._scanner:
            self._scanner.stop()
