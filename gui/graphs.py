from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtGui import QColor

class HealthGraphCanvas(FigureCanvas):
    """
    Matplotlib canvas for plotting health over time. Displays health percentage (0-100%).
    """

    def __init__(self, parent=None, time_window=60.0):
        """
        2D line graph of health over time.
        :param parent:
        :param time_window: float
            seconds of history to display
        """
        # Figure background
        self.fig = Figure(figsize=(5, 3), facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.setParent(parent)

        # Match Qt canvas background (fixes white edge line)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor("#1e1e1e"))
        self.setPalette(p)
        self.setStyleSheet("background-color: #1e1e1e; border: none;")

        # Axes setup
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax.ticklabel_format(style='plain', axis='y')

        # Remove spines (borders)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Remove labels and titles
        self.ax.set_xlabel("")
        self.ax.set_ylabel("Health (%)", fontsize=8, labelpad=10, color="#FFFFFF")
        self.ax.set_title("")

        # Invisible ticks (required for gridlines)
        # Y ticks for horizontal gridlines
        self.ax.set_yticks([0, 20, 40, 60, 80, 100])
        self.ax.tick_params(axis="y", length=0, labelsize=8, pad=-2, colors="#FFFFFF",)
        for label in self.ax.get_yticklabels():
            label.set_horizontalalignment("left")

        # X ticks for vertical gridlines
        x_ticks = [-60, -50, -40, -30, -20, -10, 0]
        self.ax.set_xticks(x_ticks)
        self.ax.tick_params(axis="x", length=0, labelsize=0)

        # Ensure grid draws above background
        self.ax.set_axisbelow(False)
        self.ax.patch.set_zorder(0)

        # Gridlines
        self.ax.grid(axis="x", color="#FFFFFF", alpha=0.25, linewidth=1.0, zorder=5)
        self.ax.grid(axis="y", color="#FFFFFF", alpha=0.25, linewidth=1.0, zorder=5)

        # Health line (heart-style)
        self._line, = self.ax.plot([], [], color="red", linewidth=2.5)

        # Underfill placeholder
        self._fill = None

        # Floating labels outside the graph (left + right)
        self._label_oldest = self.ax.text(
            -0.02, 0, "",
            transform=self.ax.get_yaxis_transform(),
            color="white",
            fontsize=8,
            ha="right",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#1e1e1e",
                edgecolor="red",
                linewidth=0.8,
                alpha=0.8
            )
        )

        self._label_newest = self.ax.text(
            1.02, 0, "",
            transform=self.ax.get_yaxis_transform(),
            color="white",
            fontsize=8,
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#1e1e1e",
                edgecolor="red",
                linewidth=0.8,
                alpha=0.8
            )
        )

        # Time window
        self._time_window = time_window

        # Tight layout for HUD look
        self.fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.04)

    def update_series(self, times, health_values):
        """
        Update the health time series data.

        :param times:
        :param health_values:
        :return:
        """
        if not times:
            return

        # Trim to last time_window seconds
        t_max = times[-1]
        t_min = t_max - self._time_window
        trimmed_t = []
        trimmed_h = []
        for t, h in zip(times, health_values):
            if t >= t_min:
                trimmed_t.append(t - t_max)  # shift so latest point is at 0
                trimmed_h.append(h)

        # Update line
        self._line.set_data(trimmed_t, trimmed_h)

        # Update underfill
        if self._fill:
            self._fill.remove()
            self._fill = None

        if trimmed_t:
            self._fill = self.ax.fill_between(
                trimmed_t,
                trimmed_h,
                0,
                color="red",
                alpha=0.15
            )

        # Floating labels for oldest + newest points
        if trimmed_t:
            # Oldest point
            h_old = trimmed_h[0]
            self._label_oldest.set_y(h_old)
            self._label_oldest.set_text(f"{h_old:.2f}")

            # Newest point
            h_new = trimmed_h[-1]
            self._label_newest.set_y(h_new)
            self._label_newest.set_text(f"{h_new:.2f}")

        self.ax.set_xlim(-self._time_window, 0)
        self.ax.set_ylim(0, 100)
        self.ax.figure.canvas.draw_idle()

class DpsGraphCanvas(FigureCanvas):
    """
    Matplotlib canvas for plotting d(health)/dt over time. Displays health change rate (%/s).
    """
    def __init__(self, parent=None, time_window=60.0):
        """
        2D line graph of health change rate over time.

        :param parent:
        :param time_window: float
            seconds of history to display
        """
        # Figure background
        self.fig = Figure(figsize=(5, 3), facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.setParent(parent)

        # Match Qt canvas background
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor("#1e1e1e"))
        self.setPalette(p)
        self.setStyleSheet("background-color: #1e1e1e; border: none;")

        # Axes setup
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax.ticklabel_format(style='plain', axis='y')

        # Remove spines
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Labels
        self.ax.set_xlabel("")
        self.ax.set_ylabel("DPS (%/s)", fontsize=8, labelpad=10, color="#FFFFFF")
        self.ax.set_title("")

        # Y ticks (autoscale will override)
        self.ax.set_yticks([0])
        self.ax.tick_params(axis="y", length=0, labelsize=8, pad=-15, colors="#FFFFFF")

        # X ticks
        x_ticks = [-60, -50, -40, -30, -20, -10, 0]
        self.ax.set_xticks(x_ticks)
        self.ax.tick_params(axis="x", length=0, labelsize=0)

        # Gridlines
        self.ax.set_axisbelow(False)
        self.ax.patch.set_zorder(0)
        self.ax.grid(axis="y", color="#FFFFFF", alpha=0.25, linewidth=1.0, zorder=5)
        self.ax.grid(axis="x", color="#FFFFFF", alpha=0.25, linewidth=1.0, zorder=5)

        # Line + fill
        self._line, = self.ax.plot([], [], color="cyan", linewidth=2.5)
        self._fill = None

        # Floating labels
        self._label_oldest = self.ax.text(
            -0.02, 0, "",
            transform=self.ax.get_yaxis_transform(),
            color="white",
            fontsize=8,
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e1e1e",
                      edgecolor="cyan", linewidth=0.8, alpha=0.8)
        )

        self._label_newest = self.ax.text(
            1.02, 0, "",
            transform=self.ax.get_yaxis_transform(),
            color="white",
            fontsize=8,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e1e1e",
                      edgecolor="cyan", linewidth=0.8, alpha=0.8)
        )

        self._time_window = time_window
        self.fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.04)

    def update_series(self, times, deriv_values):
        """
        Update the d(health)/dt time series data.

        :param times:
        :param deriv_values:
        :return:
        """
        if not times:
            return

        # Trim to window
        t_max = times[-1]
        t_min = t_max - self._time_window
        trimmed_t = []
        trimmed_d = []
        for t, d in zip(times, deriv_values):
            if t >= t_min:
                trimmed_t.append(t - t_max)
                trimmed_d.append(d)

        if not trimmed_t:
            return

        # Update line
        self._line.set_data(trimmed_t, trimmed_d)

        # Update fill
        if self._fill:
            self._fill.remove()
        self._fill = self.ax.fill_between(trimmed_t, trimmed_d, 0, color="cyan", alpha=0.15)

        # Floating labels
        self._label_oldest.set_y(trimmed_d[0])
        self._label_oldest.set_text(f"{trimmed_d[0]:.2f}")

        self._label_newest.set_y(trimmed_d[-1])
        self._label_newest.set_text(f"{trimmed_d[-1]:.2f}")

        # Axes limits
        self.ax.set_xlim(-self._time_window, 0)

        # Minimum Y range
        MIN_RANGE = 1.0  # %/s
        d_max = max(trimmed_d)
        upper = max(MIN_RANGE, d_max * 1.1)

        self.ax.set_ylim(0, upper)

        # Dynamic Y ticks
        num_ticks = 5
        step = upper / num_ticks
        ticks = [i * step for i in range(num_ticks + 1)]
        self.ax.set_yticks(ticks)

        self.ax.figure.canvas.draw_idle()
