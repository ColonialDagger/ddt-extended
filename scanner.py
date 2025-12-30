import csv
import cv2
import time
import threading

import numpy as np
import numpy.exceptions
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

class Scanner:

    # GMM-based color references per colorblind mode # TODO ADD OTHER COLORBLIND MODES
    COLOR_REFS = {
        "normal": {
            "weights": [
                0.19672552927738599,
                0.2067999133964609,
                0.18159396289707164,
                0.1677677631083592,
                0.24711283132072226
            ],
            "means": [
                [
                    133.5441455178054,
                    181.89059444285843
                ],
                [
                    135.98584483722124,
                    196.51254339967718
                ],
                [
                    125.16779106247647,
                    187.96421651859083
                ],
                [
                    148.47608609189092,
                    188.98030424581918
                ],
                [
                    121.54143205123121,
                    176.06325135016425
                ]
            ],
            "prec_chol": [
                [
                    [
                        0.20114274367899915,
                        -0.06273333549278755
                    ],
                    [
                        0.0,
                        0.1996764419749202
                    ]
                ],
                [
                    [
                        0.17101809786486283,
                        0.013458442044930635
                    ],
                    [
                        0.0,
                        0.2242104473695288
                    ]
                ],
                [
                    [
                        0.2086479253961739,
                        -0.056672032915156434
                    ],
                    [
                        0.0,
                        0.21600544375427622
                    ]
                ],
                [
                    [
                        0.16914375754336217,
                        0.012654298499640206
                    ],
                    [
                        0.0,
                        0.16801505026475905
                    ]
                ],
                [
                    [
                        0.2197256993491737,
                        -0.004358987341711091
                    ],
                    [
                        0.0,
                        0.20008824112973111
                    ]
                ]
            ],
            "log_dets": [
                6.4295949360210995,
                6.52231212721116,
                6.199117368056986,
                7.121416015951916,
                6.248744271572205
            ],
            "threshold": -11.842884581641233
        }
    }

    def __init__(
            self,
            colorblind_mode : str = "normal",
            health_buffer_size : int = 10,
            window_name : str = "Destiny 2",
            get_colors : bool = False,
            pixel_output : str = None,
    ) -> None:

        self.window_name = window_name
        self.t_last = time.time()
        self.pixel_output = pixel_output if pixel_output else None

        # Get resolution from Destiny 2 window
        self._detect_window_resolution()

        self.health = 0  # Measured boss health (0–1)
        self.delta_t = 1  # Time from frame capture to end of processing

        # Brightness related vars  # TODO Deprecate this or swap to CIELUV
        self.get_colors = get_colors
        self.darkest = (255, 255, 255)
        self.brightest = (0, 0, 0)

        # Per-brightness GMM params
        self.color_reference = self.COLOR_REFS[colorblind_mode]  # TODO REPLACE WITH COLORBLIND MODE

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

        # Health bar visibility + stabilization
        self.health_visible = False
        self._stabilizing_until = 0.0
        self.HEALTH_VISIBILITY_THRESHOLD = 50  # minimum mask pixels to consider bar visible
        self.STABILIZATION_TIME = 0.20  # seconds after bar reappears

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

        def worker():
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
                    self._last_cropped_img = cropped

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
                    prev_fraction = self._last_health_fraction or new_health_fraction
                    self._update_phase_state(now, prev_fraction, new_health_fraction)
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
        Robust phase detection with:
        - health bar visibility detection
        - stabilization window after reappearing
        - no false phase starts from bar disappearing/reappearing
        - no DPS spikes from 0→N or N→0 transitions
        """

        # 1. Determine if the health bar is visible
        visible = self._is_health_bar_visible(
            cropped_img=self._last_cropped_img,
            neg_mask=self.neg_mask
        )

        # Bar disappeared → end phase immediately
        if not visible:
            # Bar disappeared
            self.health_visible = False

            # If a phase is active, KEEP IT ACTIVE
            if self.phase_active:
                # Do not update damage or end the phase
                return

            # If no phase is active, just reset tracking
            self._last_health_fraction = None
            return

        # Bar just reappeared → start stabilization window
        if visible and not self.health_visible:
            self.health_visible = True
            self._stabilizing_until = now + self.STABILIZATION_TIME
            self._last_health_fraction = current_health
            return

        # Still stabilizing → ignore changes
        if now < self._stabilizing_until:
            self._last_health_fraction = current_health
            return

        # 2. Compute real damage
        damage = prev_health - current_health
        if abs(damage) < 1e-6:
            damage = 0.0

        # 3. Phase start
        if not self.phase_active and damage > self.PHASE_MIN_DAMAGE_FRACTION:
            self.phase_active = True
            self.phase_start_time = now
            self.phase_start_health = prev_health
            self.phase_last_damage_time = now
            self.phase_time_history.clear()
            self.phase_dps_history.clear()
            return

        # 4. Phase continues
        if self.phase_active:

            # Update last damage time
            if damage > self.PHASE_MIN_DAMAGE_FRACTION:
                self.phase_last_damage_time = now

            # Phase ends after idle timeout
            if now - self.phase_last_damage_time > self.PHASE_IDLE_TIMEOUT:
                self.phase_active = False
                return

            # Update DPS curve only if bar is visible
            if visible:
                elapsed = now - self.phase_start_time
                if elapsed > 0:
                    total_damage = max(0.0, self.phase_start_health - current_health)
                    dps = total_damage / elapsed
                    self.phase_time_history.append(elapsed)
                    self.phase_dps_history.append(dps)

    def _is_health_bar_visible(self, cropped_img, neg_mask) -> bool:
        """
        Determines whether the health bar is visible based on mask pixel count
        and basic sanity checks.
        """
        # Count mask pixels
        mask_pixels = np.count_nonzero(neg_mask)

        # If too few pixels, bar is not visible
        if mask_pixels < self.HEALTH_VISIBILITY_THRESHOLD:
            return False

        # If the cropped region is extremely dark or bright, it's probably fading
        mean_val = cropped_img.mean()
        if mean_val < 5 or mean_val > 250:
            return False

        return True

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

        # Convert to LUV
        luv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LUV).astype(np.float32)

        # Extract u*, v* only
        u = luv[..., 1]
        v = luv[..., 2]

        # TODO micro-optimization: flip bar vertically since it curves down -> less checks when it goes col by col?

        # Pass u, v into the GMM
        gmm_mask = Scanner._gmm_mask(color_reference, u, v)

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
    def _gmm_mask(ref, c0, c1):
        """
        Vectorized GMM classifier for 2‑channel input (u*, v*).
        """

        # Stack into (H, W, 2)
        x = np.stack([c0, c1], axis=-1).astype(np.float64)

        weights = np.asarray(ref["weights"], dtype=np.float64)
        means = np.asarray(ref["means"], dtype=np.float64)
        prec_chol = np.asarray(ref["prec_chol"], dtype=np.float64)
        log_dets = np.asarray(ref["log_dets"], dtype=np.float64)
        threshold = float(ref["threshold"])

        log_prob = np.full(x.shape[:2], -np.inf, dtype=np.float64)

        for w, mu, L, log_det in zip(weights, means, prec_chol, log_dets):
            diff = x - mu  # (H, W, 2)

            # 2D Cholesky transform
            y0 = L[0, 0] * diff[..., 0] + L[0, 1] * diff[..., 1]
            y1 = L[1, 0] * diff[..., 0] + L[1, 1] * diff[..., 1]

            md = y0 * y0 + y1 * y1

            lp = (
                    -0.5 * md
                    + np.log(w)
                    - 1.0 * np.log(2 * np.pi)  # 2D Gaussian → -1 * log(2π)
                    + log_det
            )

            log_prob = np.logaddexp(log_prob, lp)

        return log_prob >= threshold


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