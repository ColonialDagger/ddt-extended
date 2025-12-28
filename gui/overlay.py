import ctypes
from ctypes import wintypes

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QMainWindow

from gui.window_tracker import WindowTracker

user32 = ctypes.WinDLL("user32", use_last_error=True)

GetForegroundWindow = user32.GetForegroundWindow
GetForegroundWindow.restype = wintypes.HWND

GetWindowLong = user32.GetWindowLongW
SetWindowLong = user32.SetWindowLongW
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080


class OverlayWindow(QMainWindow):
    """
    Overlay window class for displaying text over the Destiny 2 game window.
    """

    def __init__(self):
        """
        Initialize the overlay window.
        """
        super().__init__()

        # Transparent, click-through, frameless overlay
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |  # Always on top (simple, robust)
            Qt.WindowTransparentForInput |
            Qt.NoDropShadowWindowHint
        )

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # Initial size (will be overridden by tracking)
        self.setGeometry(200, 200, 400, 200)

        # Text content
        self.text_lines = []
        self.font = QFont("Segoe UI", 20, QFont.Bold)

        # Window tracker
        self.tracker = WindowTracker("Destiny 2")

        # User-controlled enable/disable state
        self.user_enabled = False

        # Tracking timer (position + focus check)
        self.track_timer = QTimer(self)
        self.track_timer.timeout.connect(self._update_tracking)
        self.track_timer.start(100)  # 10 Hz, setting this too low/frequent causes Battleye to kill the game process

        # Repaint timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(50)  # 20 FPS

        # Start hidden
        self.hide()

    def _update_tracking(self):
        """
        Update overlay position and visibility based on game window focus and user setting.

        :return:
        """
        if not self.tracker.find_window():
            return  # Game not found

        # If user disabled overlay → stay hidden always
        if not self.user_enabled:
            if not self.isHidden():
                self.hide()
            return

        # User enabled overlay → only show when Destiny is focused
        if GetForegroundWindow() != self.tracker.hwnd:
            if not self.isHidden():
                self.hide()
            return
        else:
            if self.isHidden():
                self.show()

        # Update overlay position to match game window
        bounds = self.tracker.get_bounds()
        if not bounds:
            return

        x, y, w, h = bounds
        self.setGeometry(x, y, w, h)

    def set_text(self, lines):
        """
        Set the text lines to display in the overlay.

        :param lines:
        :return:
        """
        self.text_lines = lines

    def paintEvent(self, event):
        """
        Paint the overlay content.

        :param event:
        :return:
        """
        if not self.text_lines:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)

        x = 20
        y = 40

        for line in self.text_lines:
            # Outline
            pen = QPen(QColor(0, 0, 0, 255))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawText(x, y, line)

            # Fill
            painter.setPen(QColor(255, 255, 255, 255))
            painter.drawText(x, y, line)

            y += 40

    def show_overlay(self):
        """
        Show the overlay window and set user_enabled to True.

        :return:
        """
        print("Overlay: show() called")
        self.user_enabled = True
        self.show()

        # Hide from Alt-Tab
        hwnd = int(self.winId())
        exstyle = GetWindowLong(hwnd, GWL_EXSTYLE)
        SetWindowLong(hwnd, GWL_EXSTYLE, exstyle | WS_EX_TOOLWINDOW)

        print("Overlay visible:", self.isVisible())

    def hide_overlay(self):
        """
        Hide the overlay window and set user_enabled to False.

        :return:
        """
        print("Overlay: hide() called")
        self.user_enabled = False
        self.hide()


