import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

# Win32 constants
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

MONITOR_DEFAULTTONEAREST = 0x00000002

# API
SetWindowPos = user32.SetWindowPos
SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_uint
]
SetWindowPos.restype = wintypes.BOOL

MonitorFromWindow = user32.MonitorFromWindow
MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
MonitorFromWindow.restype = wintypes.HMONITOR

GetMonitorInfo = user32.GetMonitorInfoW
GetMonitorInfo.argtypes = [wintypes.HMONITOR, ctypes.POINTER(wintypes.RECT)]
GetMonitorInfo.restype = wintypes.BOOL
