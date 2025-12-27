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
            "weights": [
                0.21945167610917599,
                0.3070736697602109,
                0.4734746541306131
            ],
            "means": [
                [
                    29.51132997467159,
                    95.84627668386241,
                    150.08034803942581
                ],
                [
                    30.901996020669888,
                    135.90958305582876,
                    216.50589629062154
                ],
                [
                    41.052604440997065,
                    120.40708724523103,
                    185.78251598302043
                ]
            ],
            "covs": [
                [
                    [
                        132.3120424798885,
                        64.44233163240426,
                        25.814480378210792
                    ],
                    [
                        64.44233163240428,
                        149.6156984482194,
                        199.0312709297967
                    ],
                    [
                        25.814480378210792,
                        199.0312709297967,
                        319.9071091517782
                    ]
                ],
                [
                    [
                        107.53215236993678,
                        71.30648036259319,
                        -15.320986734172681
                    ],
                    [
                        71.30648036259319,
                        111.17446023692189,
                        61.66613176740297
                    ],
                    [
                        -15.320986734172681,
                        61.66613176740297,
                        154.51961432405946
                    ]
                ],
                [
                    [
                        287.58820441936166,
                        195.23825892261823,
                        -32.03767188375088
                    ],
                    [
                        195.23825892261823,
                        263.01377884901376,
                        135.69480433687346
                    ],
                    [
                        -32.03767188375088,
                        135.69480433687343,
                        310.8528573413548
                    ]
                ]
            ],
            "threshold": -21.341572192998754
        },
        2: {
            "weights": [
                0.31491072425048566,
                0.522236567241903,
                0.1628527085076113
            ],
            "means": [
                [
                    18.170902524960788,
                    113.17681878196191,
                    205.89714692571673
                ],
                [
                    33.26761445957971,
                    108.49556949029729,
                    171.9596663688369
                ],
                [
                    19.36144316559955,
                    77.52473959622861,
                    132.4891017431707
                ]
            ],
            "covs": [
                [
                    [
                        47.63762781258065,
                        78.16285996457769,
                        37.96412389893118
                    ],
                    [
                        78.16285996457769,
                        293.1927710607823,
                        229.86749546560608
                    ],
                    [
                        37.96412389893118,
                        229.86749546560608,
                        295.1165975469985
                    ]
                ],
                [
                    [
                        178.1008075956578,
                        153.03923701029476,
                        44.983660370191544
                    ],
                    [
                        153.03923701029476,
                        334.78855006536173,
                        302.21596287122145
                    ],
                    [
                        44.983660370191544,
                        302.2159628712214,
                        475.5222116301711
                    ]
                ],
                [
                    [
                        77.46864012644551,
                        42.22981566168178,
                        15.226156733543874
                    ],
                    [
                        42.22981566168178,
                        117.05491152446679,
                        171.41678946785723
                    ],
                    [
                        15.226156733543874,
                        171.41678946785723,
                        309.988779973155
                    ]
                ]
            ],
            "threshold": -23.054463569061983
        },
        3: {
            "weights": [
                0.3994291902259873,
                0.2315471020137814,
                0.36902370776023136
            ],
            "means": [
                [
                    26.11235800146514,
                    124.5838041131517,
                    208.9490754315581
                ],
                [
                    27.003361121192494,
                    95.78488636536032,
                    152.80018445025078
                ],
                [
                    49.385898710911746,
                    128.53520085443466,
                    186.23678849365436
                ]
            ],
            "covs": [
                [
                    [
                        74.12095396433067,
                        87.97924451209805,
                        42.202860562505464
                    ],
                    [
                        87.97924451209805,
                        291.56778068134855,
                        251.10174098936974
                    ],
                    [
                        42.202860562505464,
                        251.10174098936974,
                        324.1686590100383
                    ]
                ],
                [
                    [
                        101.68287488255417,
                        55.219286945148326,
                        34.26801072001495
                    ],
                    [
                        55.219286945148326,
                        152.58936335683606,
                        220.17863666791797
                    ],
                    [
                        34.26801072001495,
                        220.17863666791797,
                        371.7664634916873
                    ]
                ],
                [
                    [
                        157.19837977768842,
                        85.0984175278837,
                        2.720867645338456
                    ],
                    [
                        85.0984175278837,
                        230.20436154774285,
                        248.22148722804496
                    ],
                    [
                        2.7208676453384557,
                        248.22148722804496,
                        420.03198681636746
                    ]
                ]
            ],
            "threshold": -24.5381326357971
        },
        4: {
            "weights": [
                0.34972551223343795,
                0.3167355719042624,
                0.33353891586229967
            ],
            "means": [
                [
                    33.55574615298141,
                    135.9138554996624,
                    213.62871456656293
                ],
                [
                    61.69335757294958,
                    141.07839609486237,
                    192.6324503407037
                ],
                [
                    38.68575625281878,
                    110.23983369214815,
                    167.42533471900262
                ]
            ],
            "covs": [
                [
                    [
                        110.94928682438169,
                        108.54531435111903,
                        36.996735196593924
                    ],
                    [
                        108.54531435111903,
                        255.2336842243849,
                        187.0552287915105
                    ],
                    [
                        36.996735196593924,
                        187.0552287915105,
                        233.19029990275615
                    ]
                ],
                [
                    [
                        176.54704996658265,
                        83.44187327891034,
                        2.506023458787999
                    ],
                    [
                        83.44187327891034,
                        183.59839595362106,
                        211.4874976794909
                    ],
                    [
                        2.506023458787999,
                        211.48749767949084,
                        341.8345895723253
                    ]
                ],
                [
                    [
                        202.3271905993621,
                        112.88317515008941,
                        101.05919268021064
                    ],
                    [
                        112.88317515008941,
                        177.99076999209592,
                        230.21992927693933
                    ],
                    [
                        101.05919268021064,
                        230.21992927693933,
                        359.688344420184
                    ]
                ]
            ],
            "threshold": -19.256179656405017
        },
        5: {
            "weights": [
                0.2989309443368568,
                0.39141821650465475,
                0.3096508391584885
            ],
            "means": [
                [
                    44.56340011872641,
                    117.98567991328292,
                    171.47235519770257
                ],
                [
                    39.974984170247865,
                    143.3615846990793,
                    216.0677469981824
                ],
                [
                    70.5841455723777,
                    149.8123679070555,
                    198.62477535877647
                ]
            ],
            "covs": [
                [
                    [
                        206.63152307819595,
                        96.05423202879366,
                        79.24305818302463
                    ],
                    [
                        96.05423202879365,
                        164.17151797181177,
                        217.3649375423525
                    ],
                    [
                        79.24305818302463,
                        217.3649375423525,
                        334.02824865634534
                    ]
                ],
                [
                    [
                        130.50838076560137,
                        116.21659054761491,
                        40.75824385263339
                    ],
                    [
                        116.21659054761491,
                        248.1586511853911,
                        179.36226470005343
                    ],
                    [
                        40.75824385263339,
                        179.36226470005343,
                        228.7943049867048
                    ]
                ],
                [
                    [
                        172.21230915579545,
                        60.850688878412264,
                        -14.491113972676406
                    ],
                    [
                        60.850688878412264,
                        153.97154340169504,
                        182.65891991276916
                    ],
                    [
                        -14.491113972676402,
                        182.65891991276916,
                        299.4037786371324
                    ]
                ]
            ],
            "threshold": -21.801475460904946
        },
        6: {
            "weights": [
                0.3677217761077608,
                0.3062086312346503,
                0.32606959265758884
            ],
            "means": [
                [
                    44.70347611300916,
                    149.84741871632255,
                    219.75901961153107
                ],
                [
                    51.36726179430531,
                    126.28859720488157,
                    178.22465161806127
                ],
                [
                    77.59071498785237,
                    156.31228880801092,
                    202.60390609544547
                ]
            ],
            "covs": [
                [
                    [
                        149.37963387726055,
                        135.2713009179853,
                        60.36460360995971
                    ],
                    [
                        135.27130091798531,
                        250.34607831453923,
                        174.60176940312098
                    ],
                    [
                        60.36460360995971,
                        174.60176940312098,
                        199.24486322117377
                    ]
                ],
                [
                    [
                        257.64598733161614,
                        128.18871319174497,
                        113.19339538291062
                    ],
                    [
                        128.18871319174497,
                        174.1944263440628,
                        216.190522676502
                    ],
                    [
                        113.19339538291064,
                        216.190522676502,
                        312.0796405222529
                    ]
                ],
                [
                    [
                        193.89442307404366,
                        65.24631156209446,
                        -16.7859113146875
                    ],
                    [
                        65.24631156209446,
                        150.13646733335054,
                        177.3153833069145
                    ],
                    [
                        -16.785911314687503,
                        177.3153833069145,
                        289.7794297244256
                    ]
                ]
            ],
            "threshold": -18.790818049867667
        },
        7: {
            "weights": [
                0.3999465580376113,
                0.29802295767661147,
                0.3020304842857773
            ],
            "means": [
                [
                    50.18239463572058,
                    152.959570975602,
                    219.66976586549123
                ],
                [
                    57.819923596000166,
                    131.31543839199287,
                    180.3592586429002
                ],
                [
                    84.0291704878858,
                    161.56298804330604,
                    206.23610088978631
                ]
            ],
            "covs": [
                [
                    [
                        148.48468062650193,
                        118.29145517126894,
                        49.97758236600473
                    ],
                    [
                        118.29145517126894,
                        233.0502311144653,
                        164.5108614266751
                    ],
                    [
                        49.97758236600475,
                        164.5108614266751,
                        190.21268109428138
                    ]
                ],
                [
                    [
                        260.21934306990846,
                        110.63548277402113,
                        87.78133910675683
                    ],
                    [
                        110.63548277402114,
                        153.93461195407653,
                        192.55853009166205
                    ],
                    [
                        87.78133910675683,
                        192.55853009166205,
                        283.553202164417
                    ]
                ],
                [
                    [
                        182.16831546110905,
                        53.92351572933698,
                        -21.700737788676935
                    ],
                    [
                        53.92351572933698,
                        133.2585722443751,
                        156.97865302805613
                    ],
                    [
                        -21.700737788676932,
                        156.97865302805613,
                        256.7191661601756
                    ]
                ]
            ],
            "threshold": -20.2122972810717
        }
    }

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

        gmm_mask = Scanner._gmm_mask(color_reference, r, g, b)

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
    def _gmm_mask(ref, r, g, b):
        """
        Vectorized GMM classifier for entire image.
        r, g, b are float32 arrays of shape (H, W)
        Returns boolean mask of shape (H, W)
        """
        x = np.stack([r, g, b], axis=-1)  # (H, W, 3)

        # log-sum-exp accumulator
        log_prob = np.full(x.shape[:2], -np.inf, dtype=np.float64)

        for w, mu, inv_cov, log_det in zip(
                ref["weights"],
                ref["mean"],
                ref["inv_cov"],
                ref["log_det"],
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

        return log_prob >= ref["threshold"]


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