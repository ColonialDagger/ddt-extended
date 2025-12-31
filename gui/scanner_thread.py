import ctypes
from scanner import scanner
import threading
import time

from PySide6.QtCore import QObject, Signal

class ScannerThread(QObject):
    """
    Threaded wrapper around the scanner.Scanner class.
    """
    health_changed = Signal(float)
    fps_changed = Signal(float)
    delta_changed = Signal(float)

    def __init__(self, colorblind_mode="normal", buffer_size=10):
        """
        :param colorblind_mode: int
            In-game colorblind mode (1-7)
        :param buffer_size:
            Number of frames to buffer for FPS calculation
        """
        super().__init__()
        self._colorblind_mode = colorblind_mode
        self._buffer_size = buffer_size
        self._running = False
        self._thread = None
        self._scanner = None

    def start(self):
        """
        Start the scanner thread.

        :return:
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def _thread_main(self):
        """
        Main thread function.

        :return:
        """
        # WinRT requires STA apartment
        ctypes.windll.ole32.CoInitializeEx(None, 2)

        # Create scanner in this thread
        self._scanner = scanner.Scanner(self._colorblind_mode, self._buffer_size)

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
        """
        Stop the scanner thread.

        :return:
        """
        self._running = False
        if self._scanner:
            self._scanner.stop()
