import csv
import cv2
import time
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.exceptions
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

if __name__ == "__main__":
    from health_estimator import estimate_health, HealthReference
    from phase_tracker import PhaseTracker
else:
    from scanner.health_estimator import estimate_health, HealthReference
    from scanner.phase_tracker import PhaseTracker


class CaptureState:
    """
    Holds the capture agent of the health scanner.

    Does not hold the actual start/stop capture functions which are located in scanner/scanner.py for optimization.
    See scanner.Scanner.start_capture() for more details.
    """
    stop_requested: bool = False
    control: InternalCaptureControl | None = None
    last_cropped_img: Any | None = None
    delta_t: float = 1.0


@dataclass
class HealthSmoothing:
    """
    Contains buffer to smooth data reporting.
    """
    buffer: list[float] = field(default_factory=list)
    size: int = 10


class Scanner:
    """
    Scans an active Destiny 2 window to report health of a given boss.
    """


    def __init__(
            self,
            colorblind_mode: str = "normal",
            health_buffer_size: int = 10,
            window_name: str = "Destiny 2",
            get_colors: bool = False,
            pixel_output: str = None,
    ) -> None:

        self.window_name = window_name
        self.pixel_output = pixel_output if pixel_output else None

        # Get resolution from Destiny 2 window
        self.resolution = self._detect_window_resolution(window_name=self.window_name)

        # Measured boss health (0–1)
        self.health = 0

        # DEV tool to record all healthbar pixels to a CSV file.
        self.get_colors = get_colors

        # Defines reference object
        self.reference_data = HealthReference()

        # Load LUT for this colorblind mode
        lut_raw = np.fromfile(f"luts/{colorblind_mode}.bin", dtype=np.uint32)
        self.reference_data.lut = lut_raw.reshape(256, 256)

        # Define mask for pixel capture
        negative = cv2.imread(f"negatives/{self.resolution[0]}x{self.resolution[1]}_negative.png")
        self.y1, self.y2, self.x1, self.x2 = self._crop_dimensions_from_image(negative)
        self.reference_data.neg_mask = self._crop_neg_mask(negative, self.y1, self.y2, self.x1, self.x2)

        # Precompute mask pixel counts (per column) and flattened mask indices
        ys, xs = np.where(self.reference_data.neg_mask)
        self.reference_data.mask_indices = (ys, xs)  # flattened mask coordinates
        self.reference_data.mask_col_ids = xs  # column index for each mask pixel
        self.reference_data.mask_counts = np.bincount(xs, minlength=self.reference_data.neg_mask.shape[1])

        # Buffer that contains last x frames
        self.smoothing = HealthSmoothing(size=health_buffer_size)

        # Capture control state
        self.capture = CaptureState()

        # Phase tracking + visibility
        self.phase_tracker = PhaseTracker(
            min_damage_fraction=0.0005,   # ~0.05% health
            idle_timeout=5.0              # seconds without damage -> phase ends
        )

        # Time tracking
        self.last_t = time.time()
        self.dt = 0

    def start_capture(self) -> None:
        """
        Starts the capture session using the Windows Graphics Capture API. This is its own threaded process that will
        run until the stop_capture() is called.

        Don't touch anything here. This is a black hole that I got working and won't touch unless you absolutely
        have to. Every time I touch something, it breaks again.

        :return: None
        """

        def worker():
            capture = WindowsCapture(
                cursor_capture=False,
                draw_border=False,
                monitor_index=None,
                window_name=self.window_name,
            )

            # Start capture in a separate thread so start_capture() returns immediately.
            # WindowsCapture.start() runs the capture loop internally; run it on a thread.
            self.capture.stop_requested = False

            # called every time a new frame is available
            @capture.event
            def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl) -> None:
                """
                Called every time a frame is captured.

                :param frame: Frame
                    The Frame object of the last captured frame.
                :param capture_control: InternalCaptureControl
                    The capture control object.
                :return: None
                """
                try:
                    # Keep a reference to the capture control so stop_capture() can call it
                    self.capture.control = capture_control

                    self._process_frame(frame)

                    # Stop capture when requested
                    if self.capture.stop_requested:
                        capture_control.stop()

                except Exception as e:
                    print("ERROR: ", e)

            # Called when the capture item closes (usually when the window closes).
            @capture.event
            def on_closed() -> None:
                """
                Called when the window is closed.

                :return: None
                """
                print("Window has been closed.")

            capture.start()

        threading.Thread(target=worker, daemon=True).start()

    def stop_capture(self) -> None:
        """
        Stops the capture session by setting a flag.

        :return: None
        """

        # We set a flag that the capture thread checks inside on_frame_arrived. Because
        # capture_control.stop() must be called from the capture thread (the handler),
        # we don't call it directly here unless we already have a capture_control and are
        # sure it's safe to call from another thread.
        self.capture.stop_requested = True

    def get_health(self) -> float:
        """
        Returns the last measured health value.

        :return: float
            Last measured health value.
        """
        return 100 * self.health

    def get_dt(self) -> float:
        """
        Returns the last measured delta time of the calculation thread.

        :return: float
            Delta time of the calculation thread.
        """
        return self.dt

    def get_fps(self) -> float:
        """
        Returns the FPS of the calculation thread.

        :return: float
            FPS of the calculation thread.
        """
        try:
            return 1 / self.dt
        except ZeroDivisionError:
            return 9999

    def get_phase_active(self) -> bool:
        """
        Returns whether a phase is currently active.

        :return: bool
            State of phase.
        """
        return self.phase_tracker.state.active

    def get_phase_series(self) -> tuple[list[float], list[float]]:
        """
        Returns (times, dps) for the current/most recent phase.

        :return: tuple[list[float], list[float]]
            times: list of elapsed seconds since phase start.
            dps: list of DPS values at those times.
        """
        return list(self.phase_tracker.state.time_history), list(self.phase_tracker.state.dps_history)

    def _process_frame(self, frame):

        # Crop to healthbar region, drop alpha
        y1, y2, x1, x2 = self.y1, self.y2, self.x1, self.x2
        cropped = frame.frame_buffer[y1:y2, x1:x2, :-1]
        self.capture.last_cropped_img = cropped

        raw_health_fraction = estimate_health(
            cropped_img=cropped,
            ref=self.reference_data,
            min_col_fraction=0.60,  # TODO Try reducing this and see what happens. Consider Golden Gun, etc.
            edge_width=1
        )

        # Keeps buffer at the specified size and reports minimum health in order to avoid artifacts like
        # Well or Golden Gun creating bad data. Health can only ever go down, so the minimum value in the buffer
        # is reported to ensure instant measurements for data while avoiding showing inflated data points.
        # Size is determined by health_buffer_size, which is a frame count.
        buf = self.smoothing.buffer
        buf.append(raw_health_fraction)
        if len(buf) > self.smoothing.size:
            buf.pop(0)
        new_health_fraction = min(buf)  # 0–1

        # Phase tracking update (includes visibility)
        if self.phase_tracking_enabled:
            self.phase_tracker.update(
                now=time.time(),
                current_health=new_health_fraction,
                cropped_img=cropped,
                neg_mask=self.reference_data.neg_mask
            )
        else:
            # If phase tracker is disabled, freeze phase state
            self.phase_tracker.state.active = False
            self.phase_tracker.state.time_history.clear()
            self.phase_tracker.state.dps_history.clear()

        # Publish health for external callers
        self.health = new_health_fraction

        # Sends pixel data to a CSV file. Used to determine color data
        if self.pixel_output:
            self._pixels_to_csv(self.pixel_output, cropped[self.reference_data.neg_mask])

        # Timing data
        now = time.perf_counter()
        self.dt = now - self.last_t
        self.last_t = now

        return

    @staticmethod
    def _pixels_to_csv(path, pixels) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(p.tolist() for p in pixels)
        print("Data saved to csv.")

    @staticmethod
    def _crop_neg_mask(negative, health_y1: int, health_y2: int, health_x1: int, health_x2: int):
        """
        Creates the negative mask for health estimation.

        :param negative: np.ndarray (H, W, 3)
            Full-sized negative mask.
        :param health_y1: int
            Position of the top-most pixel in the mask.
        :param health_y2: int
            Position of the bottom-most pixel in the mask.
        :param health_x1: int
            Position of the left-most pixel in the mask.
        :param health_x2: int
            Position of the right-most pixel in the mask.
        :return: np.ndarray (H, W, 3)
            Cropped negative mask.
        """
        negative = negative[health_y1:health_y2, health_x1:health_x2]  # Crop image to given dimensions
        neg_mask = np.any(negative != 0, axis=2)  # Convert the cropped negative into a boolean mask
        return neg_mask

    @staticmethod
    def _crop_dimensions_from_image(negative) -> tuple[int, int, int, int]:
        """
        Grabs the crop dimensions of a negative mask.

        :param negative: np.ndarray (H, W, 3)
            The negative mask.
        :return: tuple[int, int, int, int]
            A tuple of the top, bottom, left, right of the negative mask.
        """

        try:
            mask = np.any(negative != 0, axis=2)
            ys, xs = np.where(mask)
        except numpy.exceptions.AxisError:
            raise RuntimeError("Current game resolution not yet supported. Are you fullscreen?")

        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("No white pixels found in negative mask!")

        top = ys.min()
        bottom = ys.max()
        left = xs.min()
        right = xs.max()

        return top, bottom, left, right

    @staticmethod
    def _detect_window_resolution(window_name: str, timeout: float = 0.1) -> tuple[int, int]:
        """
        Detects the resolution of the target window using Windows Graphics Capture.
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


def main():

    # Start capture (scanner thread)
    scanner_instance = Scanner(colorblind_mode="normal", get_colors=True)

    # Print the most recent frame data until user exits
    try:
        scanner_thread = threading.Thread(target=scanner_instance.start_capture, daemon=True)
        scanner_thread.start()
        while True:
            print(
                f"Health: {scanner_instance.get_health():9.6f}% | "
                f"FPS: {scanner_instance.get_fps():7.2f} | "
                f"Delta_T: {scanner_instance.get_dt() / 1000:8.3} ms"
            )
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopping...")
        scanner_instance.stop_capture()
        exit(0)


if __name__ == "__main__":
    main()
