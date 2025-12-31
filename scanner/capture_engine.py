import time
from dataclasses import dataclass, field
from typing import Any

from windows_capture import WindowsCapture, Frame, InternalCaptureControl


@dataclass
class CaptureState:
    """
    Holds the capture agent of the health scanner.
    """
    stop_requested: bool = False
    control: InternalCaptureControl | None = None
    last_cropped_img: Any | None = None
    last_time: float = field(default_factory=time.time)
    delta_t: float = 1.0


def detect_window_resolution(window_name: str, timeout: float = 0.1) -> tuple[int, int]:
    """
    Detects the resolution of the target window using Windows Graphics Capture.
    Mirrors the original _detect_window_resolution behavior.
    """
    deadline = time.time() + timeout
    searching = True
    resolution: tuple[int, int] | None = None

    while searching:
        try:
            # Every Error From on_closed and on_frame_arrived Will End Up Here
            capture = WindowsCapture(
                cursor_capture=False,
                draw_border=False,
                monitor_index=None,
                window_name=window_name,
            )

            # Called Every Time A New Frame Is Available
            @capture.event
            def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
                nonlocal resolution, searching
                h, w = frame.frame_buffer.shape[:2]
                resolution = (w, h)
                searching = False
                capture_control.stop()

            # Called When The Capture Item Closes Usually When The Window Closes, Capture
            # Session Will End After This Function Ends
            @capture.event
            def on_closed():
                raise RuntimeError("Window closed while determining resolution")

            capture.start()
        except:  # Always pass, true limiter is the timeout
            if time.time() > deadline:
                raise TimeoutError("Timed out while determining window resolution")

    if resolution is None:
        raise TimeoutError("Timed out while determining window resolution")

    return resolution
