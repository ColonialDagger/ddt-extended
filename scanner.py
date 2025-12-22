import cv2
import sys
import time
import threading

import numpy as np
import numpy.exceptions
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

class Scanner:

    # Pixel color references based on brightness (gamma_control) level
    COLOR_REFERENCES = {
        1: ((3, 50, 97),
            (48, 133, 231)),
        2: ((5, 62, 110),
            (60, 145, 234)),
        3: ((9, 76, 125),
            (74, 158, 237)),
        4: ((13, 88, 136),
            (86, 167, 239)),
        5: ((17, 97, 144),
            (95, 173, 240)),
        6: ((21, 105, 151),
            (103, 179, 242)),
        7: ((25, 111, 157),
            (109, 183, 242)),
    }

    def __init__(
            self,
            brightness : int = 4,
            health_buffer_size : int = 10,
            window_name : str = "Destiny 2",
            get_colors : bool = False,
    ) -> None:
        """
        Initializes the scan object.

        :param brightness: int
            The ingame brightness value from 1-7.
        :param health_buffer_size: int
            The maximum size of the health buffer used to avoid bad data from spillover artifacts.
        :param window_name: str
            The name of the window
        :param get_colors: bool
            Send True if you want the program to get the extreme colors. Used for development.
        :return: Scanner
        """

        # Get resolution from Destiny 2 window
        try:
            w, h = self._detect_window_resolution(window_name=window_name, timeout=5.0)
        except TimeoutError:
            raise TimeoutError("Failed to detect Destiny 2 window resolution (timeout)")

        self.health = 0  # Measured boss health
        self.delta_t = 1  # Time from frame capture to end of processing

        # Brightness related vars
        self.get_colors = get_colors
        self.darkest = (255, 255, 255)  # Used to get extreme pixel values on the health bar
        self.brightest = (0, 0, 0)
        self.dark = self.COLOR_REFERENCES[brightness][0]  # Used for actual health estimation
        self.light = self.COLOR_REFERENCES[brightness][1]

        # Define mask for pixel capture
        negative = cv2.imread(f"negatives/{w}x{h}_negative.png")
        self.y1, self.y2, self.x1, self.x2 = self._grab_crop_dimensions(negative)
        self.neg_mask = self._grab_neg_mask(negative, self.y1, self.y2, self.x1, self.x2)

        # Buffer that contains last x frames
        self.health_buffer = []
        self.health_buffer_size = health_buffer_size

        # Image capture related vars
        self._stop_requested = False  # Flag used to request the capture loop stop from other threads
        self._capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            monitor_index=None,
            window_name=window_name,
        )
        self._capture_thread = threading.Thread(target=self._capture.start, daemon=True)

    def start_capture(self) -> None:
        """
        Starts the capture session using the Windows Graphics Capture API. This is its own threaded process that will
        run until the stop_capture() is called.

        Don't touch anything here. This is a black hole that I got working and won't touch unless you absolutely
        have to. Every time I touch something, it breaks again.

        :return: None
        """

        # called every time a new frame is available
        @self._capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl) -> None:
            """
            Called every time a frame is captured.

            :param frame: Frame
                The Frame object of the last captured frame.
            :param capture_control: InternalCaptureControl
                The capture control object.
            :return: None
            """

            # Keep a reference to the capture control so stop_capture() can call it
            self._capture_control = capture_control

            # Run capture processes
            self._on_capture(frame)

            # Stop capture when requested
            if self._stop_requested:
                capture_control.stop()

        # Called when the capture item closes (usually when the window closes).
        @self._capture.event
        def on_closed() -> None:
            """
            Called when the window is closed.

            :return: None
            """
            print("Window has been closed")

        # Start capture in a separate thread so start_capture() returns immediately.
        # WindowsCapture.start() runs the capture loop internally; run it on a thread.
        self._stop_requested = False
        self._capture_thread.start()

    def stop_capture(self) -> None:
        """
        Stops the capture session by setting a flag.

        :return: None
        """

        # We set a flag that the capture thread checks inside on_frame_arrived. Because
        # capture_control.stop() must be called from the capture thread (the handler),
        # we don't call it directly here unless we already have a capture_control and are
        # sure it's safe to call from another thread.
        self._stop_requested = True

        # If we already have a capture_control reference and we know it's thread-safe to call,
        # attempt to stop immediately. Otherwise, the capture thread will stop on next frame.

        if getattr(self, "_capture_control", None):
            try:
                self.capture_control.stop()
            except Exception:
                # ignore: the capture thread will stop when it next checks the flag
                pass

        # Wait for the capture thread to finish (short timeout so caller isn't blocked too long).
        if getattr(self, "_capture_thread", None):
            self._capture_thread.join(timeout=5)

        # Clear internal references
        self._capture_thread = None
        self._capture = None
        self._capture_control = None
        self._stop_requested = False

    def _on_capture(self, frame: Frame) -> None:
        """
        Called when a capture occurs.

        :param frame: windows_capture.Frame
            The Frame object of the last captured frame.
        :return: None
        """
        # Benchmarking data
        t_start = time.time()

        # Crop to healthbar region, drop alpha
        y1, y2, x1, x2 = self.y1, self.y2, self.x1, self.x2
        cropped = frame.frame_buffer[y1:y2, x1:x2, :-1]

        raw_health = self._estimate_health_optimized(
            cropped_img=cropped,
            neg_mask=self.neg_mask,
            dark=self.dark,
            light=self.light,
            min_col_fraction=0.60,  # TODO Try reducing this and see what happens. Consider Golden Gun, etc.
            edge_width=1
        )

        # Keeps buffer at the specified size and reports minimum health in order to avoid artifacts like
        # Well or Golden Gun creating bad data. Health can only ever go down, so the minimum value in the buffer
        # is reported to ensure instant measurements for data while avoiding showing inflated data points.
        # Size is determined by health_buffer_size, which is a frame count.
        self.health_buffer.append(raw_health)
        if len(self.health_buffer) > self.health_buffer_size:
            self.health_buffer.pop(0)
        self.health = min(self.health_buffer)

        # Get brightest and darkest pixels over a runtime. Used to get color references for lookup table.
        if self.get_colors:
            self._get_darkest_and_brightest_pixels(cropped, neg_mask=self.neg_mask)

        # Benchmarking data
        self.delta_t = time.time() - t_start

    def get_health(self) -> float:
        """
        Returns the last measured health value.

        :return: float
            Last measured health value.
        """
        return 100 * self.health

    def get_delta_t(self) -> float:
        """
        Returns the last measured delta time of the calculation thread.

        :return: float
            Delta time of the calculation thread.
        """
        return self.delta_t

    def get_fps(self) -> float:
        """
        Returns the FPS of the calculation thread.

        :return: float
            FPS of the calculation thread.
        """
        try:
            return 1 / self.delta_t
        except ZeroDivisionError:
            return sys.float_info.max

    def _get_darkest_and_brightest_pixels(self, cropped_img, neg_mask=None):
        """
        Returns the darkest and brightest RGB pixels in the healthbar region

        :param cropped_img: np.ndarray (H, W, 3)
            The cropped healthbar image (RGB).
        :param neg_mask: np.ndarray (H, W), optional
            Negative mask indicating which pixels belong to the healthbar. If none, all pixels are scanned.
        :return:
        """

        if neg_mask is not None:
            # Flatten only masked pixels
            mask_flat = neg_mask.ravel()
            r = cropped_img[..., 0].ravel()[mask_flat]
            g = cropped_img[..., 1].ravel()[mask_flat]
            b = cropped_img[..., 2].ravel()[mask_flat]
        else:
            # Use all pixels
            r = cropped_img[..., 0].ravel()
            g = cropped_img[..., 1].ravel()
            b = cropped_img[..., 2].ravel()

        # Compute min/max per channel
        try:
            self.darkest = (
                min([int(r.min()), self.darkest[0]]),
                min([int(g.min()), self.darkest[1]]),
                min([int(b.min()), self.darkest[2]])
            )
            self.brightest = (
                max([int(r.max()), self.brightest[0]]),
                max([int(g.max()), self.brightest[1]]),
                max([int(b.max()), self.brightest[2]])
            )
        except AttributeError:
            self.darkest = (
                int(r.min()),
                int(g.min()),
                int(b.min())
            )
            self.brightest = (
                int(r.max()),
                int(g.max()),
                int(b.max())
            )

        return self.darkest, self.brightest

    @staticmethod
    def _detect_window_resolution(window_name: str, timeout: float = 5.0) -> tuple[int, int]:
        """
        Capture a single frame to determine the window resolution.

        :param window_name: str
            The name of the window to capture.
        :param timeout: float
            Time to wait for a frame before raising TimeoutError.
        :return: tuple[int, int]
            Returns a tuple with the resolution data.
        """
        frame_event = threading.Event()
        result = {"size": None, "exception": None}

        wc = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            monitor_index=None,
            window_name=window_name,
        )

        # Handler for the first frame
        @wc.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl) -> None:
            try:
                h, w = frame.frame_buffer.shape[:2]
                # frame_buffer has shape (H, W, 4) typically (RGBA). We report (W, H).
                result["size"] = (w, h)
            except Exception as e:
                result["exception"] = e
            finally:
                # Stop the capture (must be called from capture thread)
                try:
                    capture_control.stop()
                except Exception:
                    pass
                frame_event.set()

        # Closed handler (in case window instantly closes)
        @wc.event
        def on_closed() -> None:
            # ensure the waiting thread doesn't hang
            if not frame_event.is_set():
                result["exception"] = RuntimeError("Capture closed before a frame arrived")
                frame_event.set()

        # Run capture.start() on its own thread so this function can wait synchronously
        capture_thread = threading.Thread(target=wc.start, daemon=True)
        capture_thread.start()

        # Wait for a frame (or timeout)
        finished = frame_event.wait(timeout=timeout)
        # Give the capture thread a small moment to clean up
        capture_thread.join(timeout=1.0)

        # Ensure WindowsCapture resources are released if possible
        try:
            # If wc exposes a stop/close method we try to call it; ignore if not present.
            if hasattr(wc, "stop") and callable(getattr(wc, "stop")):
                wc.stop()
        except Exception:
            pass

        if not finished:
            raise TimeoutError("Timed out waiting for a frame from windows_capture")

        if result.get("exception"):
            raise result["exception"]

        if result["size"] is None:
            raise RuntimeError("Failed to determine window frame size")

        return result["size"]

    @staticmethod
    def _grab_neg_mask(negative, health_y1:int, health_y2:int, health_x1:int, health_x2:int):
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
    def _grab_crop_dimensions(negative) -> tuple[int, int, int, int]:
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
    def _estimate_health(
            cropped_img : np.ndarray,
            neg_mask : np.ndarray,
            dark : tuple,
            light : tuple,
            min_col_fraction : float =0.60,
            edge_width : int =2
    ) -> float:
        """
        Hybrid health estimation for D2 boss health bars (right -> left scan).

        Core idea:

        - The active health bar is always the rightmost column (normal health first, then Final Stand(s)).

        - We scan from right to left and find the first column that is "full enough" as per a given threshold.

        - All columns before that point (to the left) are assumed fully healthy.

        - Around the detected edge, perform refine pixel-by-pixel scanning to count health and not health pixels.

        - Add the count of scanned healthy pixels near the detected edge to the count of assumed health pixels.

        :param cropped_img: np.ndarray (H, W, 3)
            A cropped image of the D2 health bar only.
        :param neg_mask: np.ndarray (H, W, 3)
            A pre-specified negative mask of the same size as cropped_img
        :param dark: tuple(int, int, int)
            The dark most allowable pixel
        :param light: tuple(int, int, int)
            The light most allowable pixel
        :param min_col_fraction: float
            The minimum number of pixels in a col. to be "healthy" before scanning the next col.
        :param edge_width: int
            The width from the last healthy column that should be scanned in each direction.
        :return:
        """

        h, w, _ = cropped_img.shape

        # Per-pixel color match. Avoids allocating intermediate arrays by comparing channels directly.
        r = cropped_img[..., 0]
        g = cropped_img[..., 1]
        b = cropped_img[..., 2]

        dr, dg, db = dark
        lr, lg, lb = light

        color_in_range = (
                (r >= dr) & (r <= lr) &
                (g >= dg) & (g <= lg) &
                (b >= db) & (b <= lb)
        )

        # Healthy pixels = inside mask AND within color range
        healthy_mask = color_in_range & neg_mask

        # Precompute column counts. These are 1D arrays of length w.
        mask_counts = np.count_nonzero(neg_mask, axis=0)
        healthy_counts = np.count_nonzero(healthy_mask, axis=0)

        # Precompute thresholds to avoid division inside the loop.
        thresholds = mask_counts * min_col_fraction

        # Local references for speed inside the loop.
        hc = healthy_counts
        thr = thresholds

        # Early exit for full-health case
        # If the rightmost column is full, we don't need to scan anything.
        if hc[w - 1] >= thr[w - 1] and mask_counts[w - 1] > 0:
            first_full_col = w - 1
        else:
            # Scan right -> left for the first full column
            first_full_col = -1
            for col in range(w - 1, -1, -1):
                total = mask_counts[col]
                if total == 0:
                    continue  # no bar pixels here

                if hc[col] >= thr[col]:
                    first_full_col = col
                    break

        # Fallback if no full column found
        if first_full_col < 0:
            total = mask_counts[w - 1]
            if total == 0:
                return 0.0
            return hc[w - 1] / total

        # Count ALL mask pixels BEFORE the first full column
        if first_full_col > 0:
            full_columns_count = mask_counts[:first_full_col].sum()
        else:
            full_columns_count = 0

        # Pixel-level refinement around the edge
        edge_start = max(0, first_full_col - edge_width)
        edge_end = min(w, first_full_col + edge_width + 1)

        # Vectorized slice instead of Python loops
        edge_healthy = healthy_counts[edge_start:edge_end].sum()

        # Combine counts
        total_bar_pixels = mask_counts.sum()
        total_healthy = full_columns_count + edge_healthy

        health_fraction = total_healthy / total_bar_pixels
        return min(1.0, max(0.0, health_fraction))

    @staticmethod
    def _estimate_health_optimized(
            cropped_img : np.ndarray,
            neg_mask : np.ndarray,
            dark : tuple,
            light : tuple,
            min_col_fraction : float =0.60,
            edge_width : int =2
    ) -> float:
        """
        Hybrid health estimation for D2 boss health bars (right -> left scan).
        Optimized version of Scanner.estimate_health_hybrid.

        Core idea:

        - The active health bar is always the rightmost column (normal health first, then Final Stand(s)).

        - We scan from right to left and find the first column that is "full enough" as per a given threshold.

        - All columns before that point (to the left) are assumed fully healthy.

        - Around the detected edge, perform refine pixel-by-pixel scanning to count health and not health pixels.

        - Add the count of scanned healthy pixels near the detected edge to the count of assumed health pixels.

        :param cropped_img: np.ndarray (H, W, 3)
            A cropped image of the D2 health bar only.
        :param neg_mask: np.ndarray (H, W, 3)
            A pre-specified negative mask of the same size as cropped_img
        :param dark: tuple(int, int, int)
            The dark most allowable pixel
        :param light: tuple(int, int, int)
            The light most allowable pixel
        :param min_col_fraction: float
            The minimum number of pixels in a col. to be "healthy" before scanning the next col.
        :param edge_width: int
            The width from the last healthy column that should be scanned in each direction.
        :return:
        """

        h, w, _ = cropped_img.shape

        # Step 0: Fast per-pixel color match (fused, minimal temporaries)
        r = cropped_img[..., 0]
        g = cropped_img[..., 1]
        b = cropped_img[..., 2]

        dr, dg, db = dark
        lr, lg, lb = light

        # Fuse comparisons to reduce temporary arrays
        in_range = (
                (r >= dr) & (r <= lr) &
                (g >= dg) & (g <= lg) &
                (b >= db) & (b <= lb)
        )

        # Healthy = in-range AND inside mask
        healthy_mask = np.bitwise_and(in_range, neg_mask)

        # Step 1: Vectorized column counts
        mask_counts = np.count_nonzero(neg_mask, axis=0)
        healthy_counts = np.count_nonzero(healthy_mask, axis=0)

        # Precompute thresholds
        thresholds = mask_counts * min_col_fraction

        # Local bindings for speed
        hc = healthy_counts
        thr = thresholds
        mc = mask_counts

        # Step 1.1: Fast full-health early exit
        last_col = w - 1
        if mc[last_col] > 0 and hc[last_col] >= thr[last_col]:
            first_full_col = last_col
        else:
            # Step 1.2: Scan right → left for first full column
            first_full_col = -1
            for col in range(last_col, -1, -1):
                if mc[col] == 0:
                    continue
                if hc[col] >= thr[col]:
                    first_full_col = col
                    break

        # Step 1.5: Fallback if no full column found
        if first_full_col < 0:
            total = mc[last_col]
            if total == 0:
                return 0.0
            return hc[last_col] / total

        # Step 2: Count all mask pixels BEFORE the first full column
        if first_full_col > 0:
            full_columns_count = mc[:first_full_col].sum()
        else:
            full_columns_count = 0

        # Step 3: Pixel-level refinement around the edge
        edge_start = max(0, first_full_col - edge_width)
        edge_end = min(w, first_full_col + edge_width + 1)

        edge_healthy = hc[edge_start:edge_end].sum()

        # Step 4: Combine counts
        total_bar_pixels = mc.sum()
        total_healthy = full_columns_count + edge_healthy

        health_fraction = total_healthy / total_bar_pixels
        return min(1.0, max(0.0, health_fraction))

def main():

    # Start capture (scanner thread)
    try:
        scanner_instance = Scanner(brightness=4, get_colors=True)
    except RuntimeError:
        print("The window is not open!")
        exit(1)

    # Print the most recent frame data until user exits
    try:
        scanner_instance.start_capture()
        while True:
            # print(f"Health: {scanner_instance.get_health():.6f}% | FPS: {scanner_instance.get_fps()\
            #     } | Delta_T: {scanner_instance.get_delta_t():.6f} | Darkest: {scanner_instance.darkest\
            #     } | Brightest: {scanner_instance.brightest}")
            print(
                f"Health: {scanner_instance.get_health():9.6f}% | "
                f"FPS: {scanner_instance.get_fps():7.2f} | "
                f"Delta_T: {scanner_instance.get_delta_t()/1000:8.3} ms | "
                f"Darkest: {scanner_instance.darkest} | "
                f"Brightest: {scanner_instance.brightest}"
            )

    except KeyboardInterrupt:
        print("Stopping...")
        scanner_instance.stop_capture()
        input("Press any key to exit...")
        exit(0)

if __name__ == "__main__":
    main()