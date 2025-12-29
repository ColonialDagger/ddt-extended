import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout

class HealthGraphWidget(QWidget):
    def __init__(self, parent=None, time_window=60.0):
        super().__init__(parent)

        self._time_window = time_window

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget(background="#1e1e1e")
        layout.addWidget(self.plot)

        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "Health (%)", color="white")
        self.plot.setMouseEnabled(x=False, y=False)

        ax_left = self.plot.getAxis("left")
        ax_left.setPen(pg.mkPen(color="white"))
        ax_left.setTextPen(pg.mkPen(color="white"))
        ax_left.setTicks([
            [(0, "0"), (20, "20"), (40, "40"), (60, "60"), (80, "80"), (100, "100")],
            [(10, "10"), (30, "30"), (50, "50"), (70, "70"), (90, "90")],
        ])

        ax_bottom = self.plot.getAxis("bottom")
        ax_bottom.setPen(pg.mkPen(color="white"))
        ax_bottom.setTextPen(pg.mkPen(color="white"))

        self.curve = self.plot.plot([], [], pen=pg.mkPen("red", width=2.5))
        self.zero_curve = pg.PlotCurveItem([], [], pen=None)
        self.fill = pg.FillBetweenItem(
            self.curve,
            self.zero_curve,
            brush=pg.mkBrush(255, 0, 0, 60)
        )
        self.plot.addItem(self.fill)

        self.plot.setXRange(-self._time_window, 0, padding=0)
        self.plot.setYRange(0, 100, padding=0)
        self.plot.enableAutoRange(axis='x', enable=False)
        self.plot.enableAutoRange(axis='y', enable=False)

        # Force consistent left-axis width
        plot_item = self.plot.getPlotItem()
        left_axis = self.plot.getPlotItem().getAxis("left")
        left_axis.setWidth(40)

    def update_series(self, times, values):
        if not times:
            return

        # Trim to window
        t_max = times[-1]
        t_min = t_max - self._time_window

        trimmed_t = []
        trimmed_h = []

        for t, h in zip(times, values):
            if t >= t_min:
                trimmed_t.append(t - t_max)
                trimmed_h.append(h)

        # Update line
        self.curve.setData(trimmed_t, trimmed_h)

        # Update fill (zero baseline)
        self.zero_curve.setData(trimmed_t, [0] * len(trimmed_t))

class DpsGraphWidget(QWidget):
    def __init__(self, parent=None, time_window=60.0):
        super().__init__(parent)

        self._time_window = time_window
        self._ymax = 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget(background="#1e1e1e")
        layout.addWidget(self.plot)

        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "DPS (%/s)", color="white")
        self.plot.setMouseEnabled(x=False, y=False)

        ax_left = self.plot.getAxis("left")
        ax_left.setPen(pg.mkPen(color="white"))
        ax_left.setTextPen(pg.mkPen(color="white"))

        ax_bottom = self.plot.getAxis("bottom")
        ax_bottom.setPen(pg.mkPen(color="white"))
        ax_bottom.setTextPen(pg.mkPen(color="white"))

        self.curve = self.plot.plot([], [], pen=pg.mkPen("cyan", width=2.5))
        self.zero_curve = pg.PlotCurveItem([], [], pen=None)
        self.fill = pg.FillBetweenItem(
            self.curve,
            self.zero_curve,
            brush=pg.mkBrush(0, 255, 255, 60)
        )
        self.plot.addItem(self.fill)

        self.plot.setXRange(-self._time_window, 0, padding=0)
        self.plot.setYRange(0, self._ymax, padding=0)
        self.plot.enableAutoRange(axis='x', enable=False)

        # Force consistent left-axis width
        plot_item = self.plot.getPlotItem()
        left_axis = self.plot.getPlotItem().getAxis("left")
        left_axis.setWidth(40)

    def update_series(self, times, values):
        if not times:
            return

        # Trim to window
        t_max = times[-1]
        t_min = t_max - self._time_window

        trimmed_t = []
        trimmed_d = []

        for t, d in zip(times, values):
            if t >= t_min:
                trimmed_t.append(t - t_max)
                trimmed_d.append(d)

        if not trimmed_t:
            return

        # Update line
        self.curve.setData(trimmed_t, trimmed_d)

        # Update fill
        self.zero_curve.setData(trimmed_t, [0] * len(trimmed_t))

        # Expand Y range only when needed
        d_max = max(trimmed_d)
        if d_max > self._ymax:
            self._ymax = d_max * 1.1
            self.plot.setYRange(0, self._ymax, padding=0)

