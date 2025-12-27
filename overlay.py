import ctypes
from ctypes import wintypes
import threading
import time

# Win32 DLLs
user32   = ctypes.windll.user32
gdi32    = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# Core Win32 constants
WM_DESTROY   = 0x0002
WM_PAINT     = 0x000F
WM_NCCREATE  = 0x0081
WM_HOTKEY    = 0x0312

SW_HIDE      = 0
SW_SHOW      = 5

WS_POPUP     = 0x80000000
WS_VISIBLE   = 0x10000000

WS_EX_LAYERED     = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST     = 0x00000008
WS_EX_TOOLWINDOW  = 0x00000080

LWA_COLORKEY = 0x00000001

TRANSPARENT_COLOR = 0x00FF00FF  # magenta color key

# Overlay mode constants
MODE_FULLSCREEN = 1
MODE_WINDOWED   = 2
MODE_HIDDEN     = 3

# Hotkey constants
MOD_CONTROL = 0x0002
MOD_SHIFT   = 0x0004
VK_Q        = 0x51

# Win32 type definitions
# LRESULT   = ctypes.c_long  # 32-bit size
LRESULT = ctypes.c_ssize_t  # 64-bit size
HCURSOR   = ctypes.c_void_p
HICON     = ctypes.c_void_p
HBRUSH    = ctypes.c_void_p
HINSTANCE = ctypes.c_void_p
LPWSTR    = ctypes.c_wchar_p

# Callback types
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)

EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HWND,
    wintypes.LPARAM,
)

# Function signatures
user32.DefWindowProcW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
user32.DefWindowProcW.restype = LRESULT

# Win32 Structures
class PAINTSTRUCT(ctypes.Structure):
    """
    PAINTSTRUCT structure for WM_PAINT handling
    """
    _fields_ = [
        ("hdc",       wintypes.HDC),
        ("fErase",    wintypes.BOOL),
        ("rcPaint",   wintypes.RECT),
        ("fRestore",  wintypes.BOOL),
        ("fIncUpdate",wintypes.BOOL),
        ("rgbReserved", ctypes.c_byte * 32),
    ]

class WNDCLASS(ctypes.Structure):
    """
    WNDCLASS structure for window class registration
    """
    _fields_ = [
        ("style",        wintypes.UINT),
        ("lpfnWndProc",  WNDPROC),
        ("cbClsExtra",   ctypes.c_int),
        ("cbWndExtra",   ctypes.c_int),
        ("hInstance",    HINSTANCE),
        ("hIcon",        HICON),
        ("hCursor",      HCURSOR),
        ("hbrBackground",HBRUSH),
        ("lpszMenuName", LPWSTR),
        ("lpszClassName",LPWSTR),
    ]

class Overlay:
    """
    Overlay class for displaying health information over Destiny 2
        1. Creates a transparent, click-through overlay window
        2. Detects Destiny 2 window and adjusts overlay position and size
        3. Updates health information in real-time
        4. Runs in a separate thread to avoid blocking the main application
    """

    def __init__(self):
        self.overlay_hwnd = None
        self.health_text = "Health: ??"
        self.running = False
        self.current_mode = MODE_HIDDEN

    # Public API
    def start(self) -> None:
        """
        Start the overlay in a separate thread

        :return:
        """
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._run_overlay, daemon=True).start()

    def stop(self) -> None:
        """
        Stop and destroy the overlay window

        :return:
        """
        self.running = False
        if self.overlay_hwnd:
            user32.DestroyWindow(self.overlay_hwnd)
            self.overlay_hwnd = None

    def set_health(self, value) -> None:
        """
        Update the health text displayed on the overlay

        :param value:
        :return:
        """
        self.health_text = f"Health: {value:.2f}%"
        if self.overlay_hwnd:
            user32.InvalidateRect(self.overlay_hwnd, None, False)

    # Win32 Window Procedure
    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        """
        Window procedure for handling overlay messages

        :param hwnd:
        :param msg:
        :param wparam:
        :param lparam:
        :return:
        """
        if msg == WM_NCCREATE:
            return 1

        if msg == WM_PAINT:
            ps = PAINTSTRUCT()
            hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))

            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            brush = gdi32.CreateSolidBrush(TRANSPARENT_COLOR)
            user32.FillRect(hdc, ctypes.byref(rect), brush)

            gdi32.SetBkMode(hdc, 1)
            gdi32.SetTextColor(hdc, 0x000000FF)
            gdi32.TextOutW(hdc, 20, 20, self.health_text, len(self.health_text))

            user32.EndPaint(hwnd, ctypes.byref(ps))
            return 0

        if msg == WM_DESTROY:
            self.running = False
            user32.PostQuitMessage(0)
            return 0

        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    # Destiny 2 Window Detection
    def _find_destiny_window(self):
        """
        Find the Destiny 2 window by enumerating all top-level windows

        :return:
        """
        found = {"hwnd": None}

        @EnumWindowsProc
        def enum_proc(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True

            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value

            if "Destiny 2" in title:
                found["hwnd"] = hwnd
                return False

            return True

        user32.EnumWindows(enum_proc, 0)
        return found["hwnd"]

    # Overlay Positioning
    def _set_overlay_fullscreen(self):
        """
        Set the overlay to cover the entire screen

        :return:
        """
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        user32.SetWindowPos(self.overlay_hwnd, -1, 0, 0, w, h, 0)

    def _set_overlay_windowed(self, rect):
        """
        Set the overlay to match the Destiny 2 window position and size

        :param rect:
        :return:
        """
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        user32.SetWindowPos(self.overlay_hwnd, -1, rect.left, rect.top, w, h, 0)

    # Mode Thread
    def _mode_thread(self):
        """
        Thread to monitor Destiny 2 window state and adjust overlay mode

        :return:
        """
        last_mode = None
        last_rect = None

        while self.running:
            hwnd = self._find_destiny_window()

            if not hwnd:
                if last_mode != MODE_HIDDEN:
                    self._hide_overlay()
                    last_mode = MODE_HIDDEN
                time.sleep(0.5)
                continue

            foreground = user32.GetForegroundWindow()
            if foreground != hwnd:
                if last_mode != MODE_HIDDEN:
                    self._hide_overlay()
                    last_mode = MODE_HIDDEN
                time.sleep(0.5)
                continue

            # Destiny is focused -> ensure overlay is topmost
            user32.SetWindowPos(
                self.overlay_hwnd,
                -1,
                0, 0, 0, 0,
                0x0002 | 0x0001 | 0x0010
            )

            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                time.sleep(0.5)
                continue

            width = rect.right - rect.left
            height = rect.bottom - rect.top

            if width <= 0 or height <= 0:
                if last_mode != MODE_HIDDEN:
                    self._hide_overlay()
                    last_mode = MODE_HIDDEN
                time.sleep(0.5)
                continue

            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)

            current_rect = (rect.left, rect.top, rect.right, rect.bottom)

            if current_rect != last_rect:
                is_fullscreen = (
                    width == screen_w and
                    height == screen_h and
                    rect.left == 0 and
                    rect.top == 0
                )

                if is_fullscreen:
                    self._show_overlay()
                    self._set_overlay_fullscreen()
                    last_mode = MODE_FULLSCREEN
                else:
                    self._show_overlay()
                    self._set_overlay_windowed(rect)
                    last_mode = MODE_WINDOWED

                last_rect = current_rect

            time.sleep(0.5)

    # Overlay Visibility
    def _hide_overlay(self):
        """
        Hide the overlay window

        :return:
        """
        if self.overlay_hwnd:
            user32.ShowWindow(self.overlay_hwnd, SW_HIDE)

    def _show_overlay(self):
        """
        Show the overlay window

        :return:
        """
        if self.overlay_hwnd:
            user32.ShowWindow(self.overlay_hwnd, SW_SHOW)

    # Overlay Main Loop
    def _run_overlay(self):
        """
        Main overlay window creation and message loop
        
        :return:
        """
        CLASS_NAME = "DDTOverlayClass"
        hInstance = kernel32.GetModuleHandleW(None)

        # Keep callback alive on the instance
        self._wndproc = WNDPROC(self._wnd_proc)

        wc = WNDCLASS()
        wc.style = 0
        wc.lpfnWndProc = self._wndproc
        wc.cbClsExtra = 0
        wc.cbWndExtra = 0
        wc.hInstance = hInstance
        wc.hIcon = None
        wc.hCursor = user32.LoadCursorW(None, 32512)
        wc.hbrBackground = None
        wc.lpszMenuName = None
        wc.lpszClassName = CLASS_NAME

        atom = user32.RegisterClassW(ctypes.byref(wc))
        if not atom:
            print("RegisterClassW failed:", kernel32.GetLastError())
            return

        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)

        hwnd = user32.CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW,
            CLASS_NAME,
            "DDT Overlay",
            WS_POPUP | WS_VISIBLE,
            0, 0, w, h,
            None, None, hInstance, None
        )

        if not hwnd:
            print("CreateWindowExW failed:", kernel32.GetLastError())
            return

        self.overlay_hwnd = hwnd

        user32.SetLayeredWindowAttributes(hwnd, TRANSPARENT_COLOR, 255, LWA_COLORKEY)
        user32.ShowWindow(hwnd, SW_SHOW)

        # Start mode thread
        threading.Thread(target=self._mode_thread, daemon=True).start()

        # Win32 message loop
        msg = wintypes.MSG()
        while self.running and user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        self.overlay_hwnd = None
