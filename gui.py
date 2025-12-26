import matplotlib.transforms as mtransforms
import threading
import time

import ttkbootstrap as tb
from ttkbootstrap.constants import *

import scanner

class DDT(tb.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("DDT Extended")
        self.geometry("1200x500")

        self.scanner_instance = None
        self.scanner_thread = None

        # Validation
        self.only_digits_vcmd = (self.register(self._validate_digits), "%P")

        # Health graph state
        self.time_history = []  # timestamps
        self.health_history = []  # health percent (0–100)
        self.last_health_pct = 100.0

        self._build_titlebar()
        self._build_main_frames()
        self._apply_theme_to_graph()

        self._update_stats()
        self._update_graph()

    def _validate_digits(self, proposed) -> bool:
        """
        Validation function to allow only digits in entry fields.

        :param proposed: float
            The proposed new value of the entry field.
        :return: bool
            True if the proposed value is valid (only digits or empty), False otherwise
        """
        return proposed == "" or proposed.isdigit()

    def _build_titlebar(self):
        """
        Build the title bar with application title and theme selector.

        :return:
        """
        titlebar = tb.Frame(self, padding=10)
        titlebar.pack(fill=X, side=TOP)

        tb.Label(
            titlebar,
            text="DDT Extended",
            font=("Segoe UI", 16, "bold")
        ).pack(side=LEFT)

        spacer = tb.Frame(titlebar)
        spacer.pack(side=LEFT, expand=True, fill=X)

        tb.Label(titlebar, text="Select Theme:").pack(side=LEFT, padx=(0, 5))

        themes = sorted(self.style.theme_names())
        self.theme_var = tb.StringVar(value=self.style.theme.name)

        theme_dropdown = tb.Combobox(
            titlebar,
            textvariable=self.theme_var,
            values=themes,
            width=20,
            state="readonly"
        )
        theme_dropdown.pack(side=LEFT)
        theme_dropdown.bind("<<ComboboxSelected>>", self._change_theme)

    def _change_theme(self, event=None):
        """
        Change the application theme.

        :param event:
        :return:
        """
        self.style.theme_use(self.theme_var.get())
        self._apply_theme_to_graph()

    def _build_main_frames(self):
        """
        Build the main frames of the application. Left is stats and graphs, right is controls and settings.

        :return:
        """
        container = tb.Frame(self, padding=10)
        container.pack(fill=BOTH, expand=True)

        # Left column
        left_column = tb.Frame(container)
        left_column.pack(side=LEFT, fill=BOTH, expand=True)

        # Stats frame
        self.left_frame = tb.Labelframe(left_column, text="Scanner Stats", padding=10)
        self.left_frame.pack(fill=X, padx=10, pady=10)

        self.health_var = tb.StringVar(value="—")
        self.fps_var = tb.StringVar(value="—")
        self.delta_var = tb.StringVar(value="—")

        tb.Label(self.left_frame, text="Health:").grid(row=0, column=0, sticky=W)
        tb.Label(self.left_frame, textvariable=self.health_var, width=15).grid(row=0, column=1, sticky=E)

        tb.Label(self.left_frame, text="FPS:").grid(row=1, column=0, sticky=W)
        tb.Label(self.left_frame, textvariable=self.fps_var, width=15).grid(row=1, column=1, sticky=E)

        tb.Label(self.left_frame, text="Delta t:").grid(row=2, column=0, sticky=W)
        tb.Label(self.left_frame, textvariable=self.delta_var, width=15).grid(row=2, column=1, sticky=E)

        # Graphs container
        graphs_container = tb.Frame(left_column)
        graphs_container.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        # Left graph: health over time
        self.graph_frame = tb.Labelframe(graphs_container, text="Boss Health Over Time", padding=0)
        self.graph_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        # Right graph: phase DPS over time
        self.graph_frame_2 = tb.Labelframe(graphs_container, text="Phase DPS Over Time", padding=0)
        self.graph_frame_2.pack(side=LEFT, fill=BOTH, expand=True)

        # Matplotlib setup
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        # Health graph
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        # DPS graph
        self.fig2 = Figure(figsize=(5, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.graph_frame_2)
        self.canvas2.get_tk_widget().pack(fill=BOTH, expand=True)

        # Right column
        self.right_column = tb.Frame(container)
        self.right_column.pack(side=RIGHT, fill=Y, padx=10, pady=10)

        # Controls
        self.controls_frame = tb.Labelframe(self.right_column, text="Controls", padding=10)
        self.controls_frame.pack(fill=X)

        # Scanner control buttons
        tb.Button(self.controls_frame, text="Start Scanner", bootstyle=SUCCESS,
                  command=self._start_scanner).pack(fill=X, pady=(0, 5))

        tb.Button(self.controls_frame, text="Stop Scanner", bootstyle=DANGER,
                  command=self._stop_scanner).pack(fill=X)

        # Settings
        self.settings_frame = tb.Labelframe(self.right_column, text="Settings", padding=10)
        self.settings_frame.pack(fill=BOTH, expand=True, pady=(15, 0))

        tb.Label(self.settings_frame, text="Brightness:").pack(anchor=W)
        self.brightness_var = tb.StringVar(value="4")
        tb.Combobox(self.settings_frame, textvariable=self.brightness_var,
                    values=["1","2","3","4","5","6","7"], width=10).pack(anchor=W, pady=(0, 10))

        tb.Label(self.settings_frame, text="Health Buffer Size:").pack(anchor=W)
        self.health_buffer_var = tb.StringVar(value="10")
        tb.Spinbox(self.settings_frame, from_=1, to=100,
                   textvariable=self.health_buffer_var, width=10).pack(anchor=W, pady=(0, 10))

    # Scanner Control
    def _start_scanner(self):
        """
        Start the scanner in a separate thread.

        :return:
        """
        if self.scanner_instance is not None:
            return

        brightness = int(self.brightness_var.get())
        health_buffer_size = int(self.health_buffer_var.get())

        self.scanner_instance = scanner.Scanner(
            brightness=brightness,
            health_buffer_size=health_buffer_size
        )

        def _run():
            try:
                if hasattr(self.scanner_instance, "start_capture"):
                    self.scanner_instance.start_capture()
                else:
                    self.scanner_instance.start()
            except Exception as e:
                print("Scanner thread error:", e)

        self.scanner_thread = threading.Thread(target=_run, daemon=True)
        self.scanner_thread.start()

    def _stop_scanner(self):
        """
        Stop the scanner and clean up.

        :return:
        """
        if self.scanner_instance is None:
            return

        try:
            if hasattr(self.scanner_instance, "stop_capture"):
                self.scanner_instance.stop_capture()
            else:
                self.scanner_instance.stop()
        except Exception as e:
            print("Error stopping scanner:", e)

        self.scanner_instance = None
        self.scanner_thread = None

    @staticmethod
    def _normalize_health_value(health):
        """
        Normalize health value to percentage (0-100).

        :param health:
        :return:
        """
        if not isinstance(health, (int, float)):
            return None
        if 0 <= health <= 1.01:
            return max(0, min(100, health * 100))
        return max(0, min(100, health))

    # Stats Update Loop
    def _update_stats(self):
        """
        Update the stats display with current scanner values.

        :return:
        """
        if self.scanner_instance:
            try:
                health = self._normalize_health_value(self.scanner_instance.get_health())
                fps = self.scanner_instance.get_fps()
                delta = self.scanner_instance.get_delta_t()

                self.health_var.set(f"{health:.2f} %")
                self.fps_var.set(f"{fps:.2f} FPS")
                self.delta_var.set(f"{delta:.2f} ms")
            except Exception as e:
                print("Scanner update error:", e)

        self.after(100, self._update_stats)

    # Theme Helpers
    @staticmethod
    def _inverse_hex_color(hex_color: str) -> str:
        """
        Invert a hex color string.

        :param hex_color: str
            The input hex color string (e.g., "#RRGGBB").
        :return: str
            The inverted hex color string.
        """

        try:
            hex_color = hex_color.lstrip("#")
            if len(hex_color) != 6:
                return "#FFFFFF"
            r = 255 - int(hex_color[0:2], 16)
            g = 255 - int(hex_color[2:4], 16)
            b = 255 - int(hex_color[4:6], 16)
            return f"#{r:02X}{g:02X}{b:02X}"
        except Exception:
            return "#FFFFFF"

    def _apply_theme_to_graph(self):
        """
        Apply the current theme colors to the graphs.

        :return:
        """
        colors = self.style.colors
        for ax, fig in [(self.ax, self.fig), (self.ax2, self.fig2)]:
            ax.set_facecolor(colors.bg)
            fig.patch.set_facecolor(colors.bg)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors=colors.fg)

        self.graph_line_color = colors.danger
        self.dps_line_color = colors.info
        self.grid_color = self._inverse_hex_color(colors.bg)

    # Graph Update Loop
    def _update_graph(self) -> None:
        """
        Update the health and DPS graphs.
        """

        now = time.time()

        # Health sample (left graph)
        if self.scanner_instance:
            try:
                raw = self.scanner_instance.get_health()
                health_pct = self._normalize_health_value(raw)

                # Missing data check BEFORE updating last_health_pct
                if health_pct == 0 and self.last_health_pct > 0:
                    health_pct = self.last_health_pct
                else:
                    self.last_health_pct = health_pct

            except Exception:
                health_pct = self.last_health_pct
        else:
            health_pct = self.last_health_pct

        # Safety double-check
        if health_pct == 0 and self.last_health_pct > 0:
            health_pct = self.last_health_pct

        # Append new sample
        self.time_history.append(now)
        self.health_history.append(health_pct)

        # Trim to last 60 seconds
        cutoff = now - 60
        while self.time_history and self.time_history[0] < cutoff:
            self.time_history.pop(0)
            self.health_history.pop(0)

        # Draw Health Graph (left)
        self.ax.clear()
        self._apply_theme_to_graph()

        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(-60, 0)

        rel_times = [t - now for t in self.time_history]

        if rel_times:
            self.ax.plot(rel_times, self.health_history,
                         color=self.graph_line_color, linewidth=2)

        self.ax.grid(True, color=self.grid_color, alpha=0.35)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Health Labels (leftmost & rightmost)
        if self.time_history:

            def offset_for_health(y):
                # High health → label above, low health → label below
                dy = -10 if y >= 80 else 10
                return mtransforms.offset_copy(self.ax.transData, fig=self.fig, x=0, y=dy, units='points')

            # Rightmost label
            right_x = rel_times[-1]
            right_y = self.health_history[-1]
            self.ax.text(
                right_x, right_y, f"{right_y:.1f}%",
                color=self.graph_line_color, fontsize=9,
                ha="right", va="center",
                transform=offset_for_health(right_y),
                bbox=dict(
                    facecolor=self.style.colors.bg,
                    edgecolor=self.graph_line_color,
                    boxstyle="round,pad=0.2",
                    linewidth=0.8, alpha=0.85
                )
            )

            # Leftmost label (only if different from rightmost)
            left_x = rel_times[0]
            left_y = self.health_history[0]
            if left_y != right_y:
                self.ax.text(
                    left_x, left_y, f"{left_y:.1f}%",
                    color=self.graph_line_color, fontsize=9,
                    ha="left", va="center",
                    transform=offset_for_health(left_y),
                    bbox=dict(
                        facecolor=self.style.colors.bg,
                        edgecolor=self.graph_line_color,
                        boxstyle="round,pad=0.2",
                        linewidth=0.8, alpha=0.85
                    )
                )

        self.canvas.draw()

        # Compute instantaneous DPS (derivative of health)
        dps_times = []
        dps_values = []

        th = self.time_history
        hh = self.health_history

        if len(th) >= 2:
            for i in range(1, len(th)):
                dt = th[i] - th[i - 1]
                if dt <= 0:
                    continue

                damage = hh[i - 1] - hh[i]
                if damage < 0:
                    damage = 0

                dps_times.append(th[i] - now)
                dps_values.append(damage / dt)

        # Draw DPS Graph (right)
        self.ax2.clear()
        self._apply_theme_to_graph()

        if not dps_values:
            self.ax2.set_xlim(0, 10)
            self.ax2.set_ylim(0, 10)
            self.ax2.grid(True, color=self.grid_color, alpha=0.35)
            self.ax2.set_xticklabels([])
            self.ax2.set_yticklabels([])
            self.canvas2.draw()
            self.after(1000, self._update_graph)
            return

        x_min, x_max = -60, 0
        y_max = max(5.0, max(dps_values) * 1.25)

        self.ax2.set_xlim(x_min, x_max)
        self.ax2.set_ylim(0, y_max)

        self.ax2.plot(dps_times, dps_values,
                      color=self.dps_line_color, linewidth=2)

        # Precompute transforms
        trans_up = mtransforms.offset_copy(self.ax2.transData, fig=self.fig2, x=0, y=10, units='points')
        trans_down = mtransforms.offset_copy(self.ax2.transData, fig=self.fig2, x=0, y=-10, units='points')

        # Rightmost DPS label (always)
        rx = dps_times[-1]
        ry = dps_values[-1]
        self.ax2.text(
            rx, ry, f"{ry:.1f}",
            color=self.dps_line_color, fontsize=9,
            ha="right", va="bottom",
            transform=trans_up,
            bbox=dict(
                facecolor=self.style.colors.bg,
                edgecolor=self.dps_line_color,
                boxstyle="round,pad=0.2",
                linewidth=0.8, alpha=0.85
            )
        )

        # Max DPS label (> 0)
        max_val = max(dps_values)
        if max_val > 0:
            i = dps_values.index(max_val)
            self.ax2.text(
                dps_times[i], max_val, f"Max: {max_val:.1f}",
                color=self.dps_line_color, fontsize=9,
                ha="left", va="bottom",
                transform=trans_up,
                bbox=dict(
                    facecolor=self.style.colors.bg,
                    edgecolor=self.dps_line_color,
                    boxstyle="round,pad=0.2",
                    linewidth=0.8, alpha=0.85
                )
            )

        # Min DPS label (> 0)
        min_val = min(dps_values)
        if min_val > 0:
            i = dps_values.index(min_val)
            self.ax2.text(
                dps_times[i], min_val, f"Min: {min_val:.1f}",
                color=self.dps_line_color, fontsize=9,
                ha="left", va="top",
                transform=trans_down,
                bbox=dict(
                    facecolor=self.style.colors.bg,
                    edgecolor=self.dps_line_color,
                    boxstyle="round,pad=0.2",
                    linewidth=0.8, alpha=0.85
                )
            )

        self.ax2.grid(True, color=self.grid_color, alpha=0.35)
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])

        self.canvas2.draw()

        self.after(1000, self._update_graph)

if __name__ == "__main__":
    app = DDT(themename="darkly")  # starting theme
    app.mainloop()