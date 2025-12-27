import csv
import cv2
import sys
import time
import threading

import numpy as np
import numpy.exceptions
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

class Scanner:

    # Pixel color references based on brightness (gamma_control) level
    # Determined by generating a scatterplot of known healthy pixel values and using linear equations
    # to create a volume in which pixels are considered to be healthy.
    COLOR_REFERENCE = {
        1: {
            "rg": {
                "m_1": 9.8,
                "b_1": 2.1,
                "m_2": 1.8,
                "b_2": 104,
                "m_3": 0,
                "b_3": 159,
                "m_4": 0.6,
                "b_4": 58.5,
                "m_5": 1.7,
                "b_5": 11,
            },
            "gb": {
                "m_1": 1.9,
                "b_1": -10,
                "m_2": 0.2,
                "b_2": 210,
                "m_3": 0,
                "b_3": 300,
                "m_4": 0.8,
                "b_4": 53.5,
                "m_5": 2.9,
                "b_5": -240,
            },
            "rb": {
                "m_1": 16.2,
                "b_1": -0.5,
                "m_2": 1,
                "b_2": 209,
                "m_3": -0.9,
                "b_3": 280,
                "m_4": 0.7,
                "b_4": 100,
                "m_5": 1.4,
                "b_5": 69,
            }
        },
        2: {
            "rg": {
                "m_1": 18,
                "b_1": -5,
                "m_2": 1.7,
                "b_2": 110,
                "m_3": -0.2,
                "b_3": 154,
                "m_4": 0.8,
                "b_4": 44,
                "m_5": 2.5,
                "b_5": -40,
            },
            "gb": {
                "m_1": 3.2,
                "b_1": -57,
                "m_2": 1.5,
                "b_2": 56,
                "m_3": 0.03,
                "b_3": 232,
                "m_4": 0.9,
                "b_4": 43,
                "m_5": 3.4,
                "b_5": -285,
            },
            "rb": {
                "m_1": 30,
                "b_1": 5,
                "m_2": 1.1,
                "b_2": 210,
                "m_3": -1.6,
                "b_3": 293,
                "m_4": 2.9,
                "b_4": -23,
                "m_5": 0.7,
                "b_5": 92,
            }
        },
        3: {
            "rg": {
                "m_1": 11,
                "b_1": -3,
                "m_2": 2,
                "b_2": 101,
                "m_3": -0.05,
                "b_3": 162,
                "m_4": 2.6,
                "b_4": -68,
                "m_5": 0.7,
                "b_5": 55,
            },
            "gb": {
                "m_1": 2.7,
                "b_1": -64,
                "m_2": 1.5,
                "b_2": 33,
                "m_3": 0.47,
                "b_3": 167,
                "m_4": 0.75,
                "b_4": 58,
                "m_5": 3.8,
                "b_5": -387,
            },
            "rb": {
                "m_1": 17,
                "b_1": 5,
                "m_2": 0.6,
                "b_2": 215,
                "m_3": -1,
                "b_3": 287,
                "m_4": 3.8,
                "b_4": -111,
                "m_5": 0.6,
                "b_5": 100,
            }
        },
        4: {
            "rg": {
                "m_1": 7.5,
                "b_1": 0,
                "m_2": 1.2,
                "b_2": 117,
                "m_3": 0,
                "b_3": 170,
                "m_4": 0.7,
                "b_4": 65,
                "m_5": 1.5,
                "b_5": 20
            },
            "gb": {
                "m_1": 4.5,
                "b_1": -220,
                "m_2": 1.3,
                "b_2": 50,
                "m_3": 0,
                "b_3": 242,
                "m_4": 0.9,
                "b_4": 45,
                "m_5": 3,
                "b_5": -285
            },
            "rb": {
                "m_1": 10.5,
                "b_1": 25,
                "m_2": 0.7,
                "b_2": 215,
                "m_3": -0.8,
                "b_3": 285,
                "m_4": 1.4,
                "b_4": 65,
                "m_5": 0.7,
                "b_5": 110
            }
        },
        5: {
            "rg": {
                "m_1": 5.8,
                "b_1": 11,
                "m_2": 0.9,
                "b_2": 130,
                "m_3": 0,
                "b_3": 178,
                "m_4": 2,
                "b_4": -53,
                "m_5": 0.6,
                "b_5": 68,
            },
            "gb": {
                "m_1": 2.1,
                "b_1": -48,
                "m_2": 1.2,
                "b_2": 56,
                "m_3": 0,
                "b_3": 244,
                "m_4": 0.9,
                "b_4": 46,
                "m_5": 3.8,
                "b_5": -444,
            },
            "rb": {
                "m_1": 9,
                "b_1": 8,
                "m_2": 0.5,
                "b_2": 215,
                "m_3": -0.7,
                "b_3": 287,
                "m_4": 1.5,
                "b_4": 47,
                "m_5": 0.4,
                "b_5": 120,
            }
        },
        6: {
            "rg": {
                "m_1": 6,
                "b_1": -19,
                "m_2": 0.9,
                "b_2": 127,
                "m_3": -0.3,
                "b_3": 203,
                "m_4": 2.4,
                "b_4": -100,
                "m_5": 0.5,
                "b_5": 80,
            },
            "gb": {
                "m_1": 3,
                "b_1": -144,
                "m_2": 1.3,
                "b_2": 32,
                "m_3": 0,
                "b_3": 244,
                "m_4": 3.2,
                "b_4": -344,
                "m_5": 0.8,
                "b_5": 57,
            },
            "rb": {
                "m_1": 8.3,
                "b_1": -20,
                "m_2": 0.4,
                "b_2": 218,
                "m_3": -0.95,
                "b_3": 310,
                "m_4": 2.2,
                "b_4": -30,
                "m_5": 0.5,
                "b_5": 122,
            }
        },
        7: {
            "rg": {
                "m_1": 4.7,
                "b_1": 3,
                "m_2": 1.05,
                "b_2": 122,
                "m_3": -0.08,
                "b_3": 190,
                "m_4": 0.4,
                "b_4": 90,
                "m_5": 2,
                "b_5": -70,
            },
            "gb": {
                "m_1": 2.2,
                "b_1": -78,
                "m_2": 1.6,
                "b_2": 53,
                "m_3": 0,
                "b_3": 243,
                "m_4": 0.81,
                "b_4": 58,
                "m_5": 10,
                "b_5": -1595,
            },
            "rb": {
                "m_1": 6,
                "b_1": 20,
                "m_2": 1,
                "b_2": 187,
                "m_3": -0.6,
                "b_3": 290,
                "m_4": 0.32,
                "b_4": 133,
                "m_5": 1.8,
                "b_5": -5,
            }
        }
    }

    import numpy as np

    # ============================
    # GMM PARAMETERS (HARDCODED)
    # ============================

    GMM_WEIGHTS = np.array([
        0.34975175562379884,
        0.31671444592806414,
        0.3335337984481371
    ])

    GMM_MEANS = np.array([
        [33.556663664424406, 135.9145693364635, 213.62876560741853],
        [61.69462946522615, 141.07796474135043, 192.6307341559729],
        [38.68581632083097, 110.24083444065437, 167.42702552333836]
    ])

    GMM_COVS = np.array([
        [[110.95936808838898, 108.55035422111852, 36.99438211891077],
         [108.55035422111852, 255.2318447409084, 187.04970243225821],
         [36.99438211891077, 187.04970243225821, 233.18462611797324]],

        [[176.53146342178883, 83.44726899264356, 2.5266994383271983],
         [83.44726899264356, 183.60602713766744, 211.49507906356587],
         [2.526699438327197, 211.4950790635658, 341.8291620825708]],

        [[202.33581379614432, 112.88674256533488, 101.06068168014548],
         [112.88674256533488, 177.99401517516054, 230.21703017738915],
         [101.06068168014548, 230.21703017738915, 359.6735032579051]]
    ])

    GMM_THRESHOLD = -19.255949242118373

    # Precompute inverses and log-dets
    GMM_INV_COVS = np.linalg.inv(GMM_COVS)
    GMM_LOG_DETS = np.log(np.linalg.det(GMM_COVS))

    def __init__(
            self,
            brightness : int = 4,
            health_buffer_size : int = 10,
            window_name : str = "Destiny 2",
            get_colors : bool = False,
            pixel_output : str = None,
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

        self.window_name = window_name
        self.t_last = time.time()
        self.pixel_output = pixel_output if pixel_output else None

        # Get resolution from Destiny 2 window
        self._detect_window_resolution()

        self.health = 0  # Measured boss health (0–1)
        self.delta_t = 1  # Time from frame capture to end of processing

        # Brightness related vars
        self.get_colors = get_colors
        self.darkest = (255, 255, 255)
        self.brightest = (0, 0, 0)
        self.color_reference = self.COLOR_REFERENCE[brightness]

        # Define mask for pixel capture
        negative = cv2.imread(f"negatives/{self.resolution[0]}x{self.resolution[1]}_negative.png")
        self.y1, self.y2, self.x1, self.x2 = self._grab_crop_dimensions(negative)
        self.neg_mask = self._grab_neg_mask(negative, self.y1, self.y2, self.x1, self.x2)

        # Buffer that contains last x frames
        self.health_buffer = []
        self.health_buffer_size = health_buffer_size

        # Capture control flag
        self._stop_requested = False  # Flag used to request the capture loop stop from other threads

        # Phase tracking state
        self._last_health_fraction = None  # last health value in 0–1
        self.phase_active = False
        self.phase_start_time = None  # wall-clock time
        self.phase_start_health = None  # health (0–1) at phase start
        self.phase_last_damage_time = None
        self.phase_time_history = []  # elapsed seconds since phase start
        self.phase_dps_history = []  # DPS = total_damage / elapsed

        # Tunables for phase detection (in health fraction units: 0–1)
        self.PHASE_MIN_DAMAGE_FRACTION = 0.0005 # ~0.05% health
        self.PHASE_IDLE_TIMEOUT = 5.0 # seconds without damage -> phase ends

    def start_capture(self) -> None:

        """
        Starts the capture session using the Windows Graphics Capture API. This is its own threaded process that will
        run until the stop_capture() is called.

        Don't touch anything here. This is a black hole that I got working and won't touch unless you absolutely
        have to. Every time I touch something, it breaks again.

        :return: None
        """

        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            monitor_index=None,
            window_name=self.window_name,
        )

        # Start capture in a separate thread so start_capture() returns immediately.
        # WindowsCapture.start() runs the capture loop internally; run it on a thread.
        self._stop_requested = False

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
                self._capture_control = capture_control

                # Crop to healthbar region, drop alpha
                y1, y2, x1, x2 = self.y1, self.y2, self.x1, self.x2
                cropped = frame.frame_buffer[y1:y2, x1:x2, :-1]

                raw_health_fraction = self._estimate_health(
                    cropped_img=cropped,
                    neg_mask=self.neg_mask,
                    color_reference=self.color_reference,
                    min_col_fraction=0.60,  # TODO Try reducing this and see what happens. Consider Golden Gun, etc.
                    edge_width=1
                )

                # Keeps buffer at the specified size and reports minimum health in order to avoid artifacts like
                # Well or Golden Gun creating bad data. Health can only ever go down, so the minimum value in the buffer
                # is reported to ensure instant measurements for data while avoiding showing inflated data points.
                # Size is determined by health_buffer_size, which is a frame count.
                self.health_buffer.append(raw_health_fraction)
                if len(self.health_buffer) > self.health_buffer_size:
                    self.health_buffer.pop(0)
                new_health_fraction = min(self.health_buffer)  # 0–1

                # Timing data
                now = time.time()
                self.delta_t = now - self.t_last
                self.t_last = now

                # Phase tracking update
                prev_fraction = self._last_health_fraction
                if prev_fraction is None:
                    prev_fraction = new_health_fraction
                self._update_phase_state(
                    now=now,
                    prev_health=prev_fraction,
                    current_health=new_health_fraction
                )
                self._last_health_fraction = new_health_fraction

                # Publish health for external callers
                self.health = new_health_fraction

                # Get brightest and darkest pixels over a runtime. Used to get color references for lookup table.
                if self.get_colors:
                    self._get_darkest_and_brightest_pixels(cropped, neg_mask=self.neg_mask)

                # Sends pixel data to a CSV file. Used to determine color data
                if self.pixel_output:
                    self._pixels_to_csv(self.pixel_output, cropped[self.neg_mask])

                # Stop capture when requested
                if self._stop_requested:
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
            print("Window has been closed")

        capture.start()

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
            return 9999

    def get_phase_active(self) -> bool:
        """
        Returns whether a phase is currently active.

        :return: bool
            State of phase.
        """
        return self.phase_active

    def get_phase_series(self) -> tuple[list[float], list[float]]:
        """
        Returns (times, dps) for the current/most recent phase.

        :return: tuple[list[float], list[float]]
            times: list of elapsed seconds since phase start.
            dps: list of DPS values at those times.
        """
        """
        Returns (times, dps) for the current/most recent phase.
        times: list of elapsed seconds since phase start.
        dps:   list of DPS values at those times.
        """
        return list(self.phase_time_history), list(self.phase_dps_history)

    def _update_phase_state(self, now: float, prev_health: float, current_health: float) -> None:
        """
        Updates phase state based on health changes.

        :param now: float
            Current time.
        :param prev_health:
            Previous health value.
        :param current_health:
            Current health value
        :return: None
        """
        """
        Runs at capture rate. Health values are fractions (0–1).
        Detects phase start/end, and updates DPS series.
        """
        damage = prev_health - current_health  # positive when boss loses health

        # Ignore tiny noise
        if abs(damage) < 1e-6:
            damage = 0.0

        # Phase start
        if not self.phase_active and damage > self.PHASE_MIN_DAMAGE_FRACTION:
            self.phase_active = True
            self.phase_start_time = now
            self.phase_start_health = prev_health
            self.phase_last_damage_time = now
            self.phase_time_history.clear()
            self.phase_dps_history.clear()
            return

        # Phase continues
        if self.phase_active:
            if damage > self.PHASE_MIN_DAMAGE_FRACTION:
                self.phase_last_damage_time = now

            # Phase end
            if self.phase_last_damage_time is not None and (now - self.phase_last_damage_time > self.PHASE_IDLE_TIMEOUT):
                self.phase_active = False
                return

            # Update DPS curve
            if self.phase_start_time is not None and self.phase_start_health is not None:
                elapsed = now - self.phase_start_time
                if elapsed > 0:
                    total_damage = self.phase_start_health - current_health
                    if total_damage < 0:
                        total_damage = 0.0
                    dps = total_damage / elapsed
                    self.phase_time_history.append(elapsed)
                    self.phase_dps_history.append(dps)

    def _detect_window_resolution(self, timeout: float = 0.1) -> None:
        timeout = time.time() + timeout
        self._searching = True
        while self._searching:
            try:
                # Every Error From on_closed and on_frame_arrived Will End Up Here
                capture = WindowsCapture(
                    cursor_capture=False,
                    draw_border=False,
                    monitor_index=None,
                    window_name=self.window_name,
                )

                # Called Every Time A New Frame Is Available
                @capture.event
                def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
                    h, w = frame.frame_buffer.shape[:2]
                    self.resolution = (w, h)
                    self._searching = False
                    capture_control.stop()

                # Called When The Capture Item Closes Usually When The Window Closes, Capture
                # Session Will End After This Function Ends
                @capture.event
                def on_closed():
                    raise RuntimeError("Window closed while determining resolution")

                capture.start()
            except:  # Always pass, true limiter is the timeout
                if time.time() > timeout:
                    raise TimeoutError("Timed out while determining window resolution")

        del self._searching  # Deletes var to avoid polluting outer scope

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
    def _pixels_to_csv(path, pixels) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(p.tolist() for p in pixels)
        print("Data saved to csv.")

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
            color_reference : dict,
            min_col_fraction : float = 0.60,
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

        # Step 0: Fast per-pixel color match (fused, minimal temporaries)
        r = cropped_img[..., 0].astype(np.float32)
        g = cropped_img[..., 1].astype(np.float32)
        b = cropped_img[..., 2].astype(np.float32)

        gmm_mask = Scanner._gmm_mask(r, g, b)

        # Healthy = in-range AND inside mask
        healthy_mask = np.bitwise_and(gmm_mask, neg_mask)

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

    @staticmethod
    def _gmm_mask(r, g, b):
        """
        Vectorized GMM classifier for entire image.
        r, g, b are float32 arrays of shape (H, W)
        Returns boolean mask of shape (H, W)
        """
        x = np.stack([r, g, b], axis=-1)  # (H, W, 3)

        # log-sum-exp accumulator
        log_prob = np.full(x.shape[:2], -np.inf, dtype=np.float64)

        for w, mu, inv_cov, log_det in zip(
                Scanner.GMM_WEIGHTS,
                Scanner.GMM_MEANS,
                Scanner.GMM_INV_COVS,
                Scanner.GMM_LOG_DETS
        ):
            diff = x - mu  # (H, W, 3)

            # Mahalanobis distance per pixel
            md = (
                    diff[..., 0] * (inv_cov[0, 0] * diff[..., 0] +
                                    inv_cov[0, 1] * diff[..., 1] +
                                    inv_cov[0, 2] * diff[..., 2]) +
                    diff[..., 1] * (inv_cov[1, 0] * diff[..., 0] +
                                    inv_cov[1, 1] * diff[..., 1] +
                                    inv_cov[1, 2] * diff[..., 2]) +
                    diff[..., 2] * (inv_cov[2, 0] * diff[..., 0] +
                                    inv_cov[2, 1] * diff[..., 1] +
                                    inv_cov[2, 2] * diff[..., 2])
            )

            lp = (
                    -0.5 * md
                    - 0.5 * log_det
                    - 1.5 * np.log(2 * np.pi)
                    + np.log(w)
            )

            log_prob = np.logaddexp(log_prob, lp)

        return log_prob >= Scanner.GMM_THRESHOLD


def main():

    # Start capture (scanner thread)
    scanner_instance = Scanner(brightness=4, get_colors=True)

    # Print the most recent frame data until user exits
    try:
        scanner_thread = threading.Thread(target=scanner_instance.start_capture, daemon=True)
        scanner_thread.start()
        while True:
            print(
                f"Health: {scanner_instance.get_health():9.6f}% | "
                f"FPS: {scanner_instance.get_fps():7.2f} | "
                f"Delta_T: {scanner_instance.get_delta_t()/1000:8.3} ms | "
                f"Darkest: {scanner_instance.darkest} | "
                f"Brightest: {scanner_instance.brightest}"
            )
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopping...")
        scanner_instance.stop_capture()
        exit(0)

if __name__ == "__main__":
    main()