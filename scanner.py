import csv
import cv2
import time
import threading

import numpy as np
import numpy.exceptions
from windows_capture import WindowsCapture, Frame, InternalCaptureControl


class Scanner:

    # GMM-based color references per brightness (1–7)
    COLOR_REFS = {
        1: {
            "weights": [
                0.4738989541733739,
                0.2179807959190418,
                0.3081202499075843
            ],
            "means": [
                [
                    41.064519053395415,
                    120.31093119745388,
                    185.65242833084125
                ],
                [
                    29.366064600300923,
                    95.79248686256315,
                    150.09620649726924
                ],
                [
                    30.965822831309346,
                    135.92562479434076,
                    216.41996708468102
                ]
            ],
            "prec_chol": [
                [
                    [
                        0.058997883502670795,
                        -0.059225718542331704,
                        0.08656640873051566
                    ],
                    [
                        0.0,
                        0.08735042501158455,
                        -0.1120926624746336
                    ],
                    [
                        0.0,
                        0.0,
                        0.09226275814025782
                    ]
                ],
                [
                    [
                        0.08748557733735275,
                        -0.043827924658833686,
                        0.1262244587183243
                    ],
                    [
                        0.0,
                        0.0915254669601501,
                        -0.3494675235162072
                    ],
                    [
                        0.0,
                        0.0,
                        0.22113444208946892
                    ]
                ],
                [
                    [
                        0.09612863820703595,
                        -0.08226273236535961,
                        0.10527330019104167
                    ],
                    [
                        0.0,
                        0.12501494088205545,
                        -0.13339137994779574
                    ],
                    [
                        0.0,
                        0.0,
                        0.11804291580076245
                    ]
                ]
            ],
            "log_dets": [
                15.302391582708346,
                12.672807549613738,
                13.11619410468198
            ],
            "threshold": -21.38796584848305
        },
        2: {
            "weights": [
                0.16535233123051485,
                0.3133298219583932,
                0.5213178468110918
            ],
            "means": [
                [
                    19.516107608216473,
                    77.67821482452628,
                    132.606461776433
                ],
                [
                    18.248910819610927,
                    113.4288494573432,
                    206.16943650525897
                ],
                [
                    33.19256869583986,
                    108.45810630796659,
                    172.0509564728349
                ]
            ],
            "prec_chol": [
                [
                    [
                        0.11245060219794654,
                        -0.05709825578024698,
                        0.15239607710287972
                    ],
                    [
                        0.0,
                        0.10297194736812008,
                        -0.34947325914794086
                    ],
                    [
                        0.0,
                        0.0,
                        0.2021147606176362
                    ]
                ],
                [
                    [
                        0.14412874390492642,
                        -0.12732788541590212,
                        0.08972949374995705
                    ],
                    [
                        0.0,
                        0.0783267612530115,
                        -0.10420259027878728
                    ],
                    [
                        0.0,
                        0.0,
                        0.10336940828039627
                    ]
                ],
                [
                    [
                        0.07455597442718294,
                        -0.06067858763006622,
                        0.07773130134428041
                    ],
                    [
                        0.0,
                        0.0705471238718872,
                        -0.11628109539717287
                    ],
                    [
                        0.0,
                        0.0,
                        0.09017934035504595
                    ]
                ]
            ],
            "log_dets": [
                12.114919099738678,
                13.506720997138066,
                15.307268781267496
            ],
            "threshold": -22.996803832325643
        },
        3: {
            "weights": [
                0.3750095418363577,
                0.2280271097874399,
                0.3969633483762024
            ],
            "means": [
                [
                    49.106562755699095,
                    128.26126791968565,
                    186.12037242760277
                ],
                [
                    26.900706719950897,
                    95.60849554917499,
                    152.53307085530832
                ],
                [
                    26.09217076245761,
                    124.62895936633147,
                    209.05708187127655
                ]
            ],
            "prec_chol": [
                [
                    [
                        0.07880984332508113,
                        -0.040565672552056044,
                        0.07520118080916409
                    ],
                    [
                        0.0,
                        0.07346043631711459,
                        -0.14143108311563837
                    ],
                    [
                        0.0,
                        0.0,
                        0.10577124921114363
                    ]
                ],
                [
                    [
                        0.0994963520037938,
                        -0.04887617648406007,
                        0.10482985995613876
                    ],
                    [
                        0.0,
                        0.09068218353891398,
                        -0.3091614805022666
                    ],
                    [
                        0.0,
                        0.0,
                        0.18792331212016852
                    ]
                ],
                [
                    [
                        0.11621015271625573,
                        -0.08745517121089151,
                        0.07685714530041297
                    ],
                    [
                        0.0,
                        0.07330709043656229,
                        -0.1171028916400231
                    ],
                    [
                        0.0,
                        0.0,
                        0.10922483498618876
                    ]
                ]
            ],
            "log_dets": [
                14.796404433579056,
                12.759499972836942,
                13.959599647101363
            ],
            "threshold": -24.537003188316536
        },
        4: {
            "weights": [
                0.34706138884256876,
                0.3378270633056831,
                0.31511154785174816
            ],
            "means": [
                [
                    33.430310015874824,
                    135.7936724356602,
                    213.59246115265302
                ],
                [
                    38.932462612870474,
                    110.4285481101295,
                    167.5594048744238
                ],
                [
                    61.64222606971795,
                    141.38444451885098,
                    193.0491856246025
                ]
            ],
            "prec_chol": [
                [
                    [
                        0.09555732855871957,
                        -0.08036546704017175,
                        0.07925521292364135
                    ],
                    [
                        0.0,
                        0.08179475450605006,
                        -0.12232358353711256
                    ],
                    [
                        0.0,
                        0.0,
                        0.12084024164682701
                    ]
                ],
                [
                    [
                        0.06981079645917379,
                        -0.05265496297174162,
                        0.051164982291088464
                    ],
                    [
                        0.0,
                        0.09330072816525076,
                        -0.22120690379937216
                    ],
                    [
                        0.0,
                        0.0,
                        0.14640399423201564
                    ]
                ],
                [
                    [
                        0.07496282848614966,
                        -0.03805916479469511,
                        0.11598835618903713
                    ],
                    [
                        0.0,
                        0.08355067919633125,
                        -0.25007477996092836
                    ],
                    [
                        0.0,
                        0.0,
                        0.1706248290754112
                    ]
                ]
            ],
            "log_dets": [
                13.929713997142686,
                13.910558731064969,
                13.682705836087596
            ],
            "threshold": -19.329929447475575
        },
        5: {
            "weights": [
                0.3924246947164071,
                0.2984454637622162,
                0.3091298415213767
            ],
            "means": [
                [
                    40.010252593791726,
                    143.38156649454962,
                    216.06105564673516
                ],
                [
                    44.542253848771146,
                    117.96492044194144,
                    171.4502527922584
                ],
                [
                    70.6185831909686,
                    149.77806372628902,
                    198.55517441327075
                ]
            ],
            "prec_chol": [
                [
                    [
                        0.08740045789037758,
                        -0.0739217057700687,
                        0.06590098429157254
                    ],
                    [
                        0.0,
                        0.08314120399059165,
                        -0.1146046797405581
                    ],
                    [
                        0.0,
                        0.0,
                        0.11582461679716757
                    ]
                ],
                [
                    [
                        0.06959937377847315,
                        -0.04248462099786643,
                        0.057218576483374105
                    ],
                    [
                        0.0,
                        0.09149058166971459,
                        -0.27169201065183857
                    ],
                    [
                        0.0,
                        0.0,
                        0.17988519647228599
                    ]
                ],
                [
                    [
                        0.07630088649763717,
                        -0.030967510806114665,
                        0.10331755121546736
                    ],
                    [
                        0.0,
                        0.08686009103112446,
                        -0.25050041291086067
                    ],
                    [
                        0.0,
                        0.0,
                        0.17677974464526994
                    ]
                ]
            ],
            "log_dets": [
                14.1602955544824,
                13.543910765270729,
                13.498756055724163
            ],
            "threshold": -21.801787346008176
        },
        6: {
            "weights": [
                0.36495958311058246,
                0.30848241708297214,
                0.3265579998064453
            ],
            "means": [
                [
                    44.576585180040574,
                    149.75067123295048,
                    219.7322044496236
                ],
                [
                    51.53205084385049,
                    126.37728654057373,
                    178.27130171108
                ],
                [
                    77.4812739498481,
                    156.4910014617365,
                    202.9046632508642
                ]
            ],
            "prec_chol": [
                [
                    [
                        0.08223286868928578,
                        -0.08035961524235544,
                        0.056082859382586926
                    ],
                    [
                        0.0,
                        0.08837719426031322,
                        -0.11847006763500524
                    ],
                    [
                        0.0,
                        0.0,
                        0.12640662415582493
                    ]
                ],
                [
                    [
                        0.0620460486489647,
                        -0.04761808805480822,
                        0.05108376184109876
                    ],
                    [
                        0.0,
                        0.09530780052053442,
                        -0.2597690135901628
                    ],
                    [
                        0.0,
                        0.0,
                        0.1793244416206463
                    ]
                ],
                [
                    [
                        0.07144776175442463,
                        -0.028737430424930297,
                        0.10948713598977164
                    ],
                    [
                        0.0,
                        0.08864898703109181,
                        -0.2751689308030539
                    ],
                    [
                        0.0,
                        0.0,
                        0.1923257617077365
                    ]
                ]
            ],
            "log_dets": [
                13.985185825785354,
                13.698161319282635,
                13.420848086789613
            ],
            "threshold": -18.801226833263755
        },
        7: {
            "weights": [
                0.2985135645034588,
                0.3030861292146153,
                0.398400306281926
            ],
            "means": [
                [
                    57.89510142660228,
                    131.3356667699067,
                    180.3539245475102
                ],
                [
                    83.93727733165444,
                    161.65176658026874,
                    206.40955737137918
                ],
                [
                    50.096884504933044,
                    152.88073222417867,
                    219.62580814262958
                ]
            ],
            "prec_chol": [
                [
                    [
                        0.061905538917324844,
                        -0.04134689845062501,
                        0.05262336151585021
                    ],
                    [
                        0.0,
                        0.0968926608469409,
                        -0.27185991206712207
                    ],
                    [
                        0.0,
                        0.0,
                        0.18716357862339764
                    ]
                ],
                [
                    [
                        0.07380704980677043,
                        -0.026661588664671013,
                        0.10350384214575879
                    ],
                    [
                        0.0,
                        0.09247220405144468,
                        -0.2708819109103308
                    ],
                    [
                        0.0,
                        0.0,
                        0.1942738155650951
                    ]
                ],
                [
                    [
                        0.08236742407261942,
                        -0.06761163860077295,
                        0.048200404962389753
                    ],
                    [
                        0.0,
                        0.08478419796145474,
                        -0.11447413520754755
                    ],
                    [
                        0.0,
                        0.0,
                        0.12739700033517334
                    ]
                ]
            ],
            "log_dets": [
                13.584138833008158,
                13.251269794123726,
                14.049316877744292
            ],
            "threshold": -20.20933736101586
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

        # Per-brightness GMM params
        self.color_reference = self.COLOR_REFS[brightness]

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
        Vectorized GMM classifier for the entire image
        :param ref: dict
            Holds keys weights, means, prec_chol, log_dets, threshold
        :param r:
            Red channel int
        :param g:
            Green channel int
        :param b:
            Blue channel int
        :return:
        """
        x = np.stack([r, g, b], axis=-1)  # (H, W, 3)

        weights = np.asarray(ref["weights"], dtype=np.float64)
        means = np.asarray(ref["means"], dtype=np.float64)
        prec_chol = np.asarray(ref["prec_chol"], dtype=np.float64)
        log_dets = np.asarray(ref["log_dets"], dtype=np.float64)
        threshold = float(ref["threshold"])

        log_prob = np.full(x.shape[:2], -np.inf, dtype=np.float64)

        for w, mu, L, log_det in zip(weights, means, prec_chol, log_dets):
            diff = x - mu  # (H, W, 3)

            y0 = L[0, 0] * diff[..., 0] + L[0, 1] * diff[..., 1] + L[0, 2] * diff[..., 2]
            y1 = L[1, 0] * diff[..., 0] + L[1, 1] * diff[..., 1] + L[1, 2] * diff[..., 2]
            y2 = L[2, 0] * diff[..., 0] + L[2, 1] * diff[..., 1] + L[2, 2] * diff[..., 2]

            md = y0 * y0 + y1 * y1 + y2 * y2

            lp = (
                -0.5 * md
                + np.log(w)
                - 1.5 * np.log(2 * np.pi)
                + log_det
            )

            log_prob = np.logaddexp(log_prob, lp)

        return log_prob >= threshold



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