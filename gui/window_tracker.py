import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

FindWindowW = user32.FindWindowW
FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
FindWindowW.restype = wintypes.HWND

GetWindowRect = user32.GetWindowRect
GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
GetWindowRect.restype = wintypes.BOOL


class WindowTracker:
    """
    Tracks the position and size of a specified window by its title.
    """

    def __init__(self, window_name="Destiny 2"):
        """
        :param window_name: The exact title of the window to track.
        """
        self.window_name = window_name
        self.hwnd = None

    def find_window(self):
        """
        Finds the window by its title and stores the handle.

        :return:
        """
        self.hwnd = FindWindowW(None, self.window_name)
        return bool(self.hwnd)

    def get_bounds(self):
        """
        Gets the current bounds of the tracked window.

        :return:
        """
        if not self.hwnd:
            return None

        rect = wintypes.RECT()
        if not GetWindowRect(self.hwnd, ctypes.byref(rect)):
            return None

        x, y = rect.left, rect.top
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        return x, y, w, h
