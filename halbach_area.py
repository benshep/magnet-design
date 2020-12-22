import sys
import numpy as np
import random
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import subprocess
halbach_exe = r"HalbachArea.exe"

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)



class App(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'HalbachArea'
        self.layout = self.controls = self.plot = None
        self.initUI()

    def appendControl(self, name, default, switch, units=None, checked=None, digits=2):
        """Add a control.
        Set checked to True or False for a checkbox, otherwise will be a label."""
        hbox = QtWidgets.QHBoxLayout()
        if checked is None:
            checkbox = QtWidgets.QLabel(name)
        else:
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.setChecked(checked)
        hbox.addWidget(checkbox)
        spinbox = QtWidgets.QDoubleSpinBox(decimals=digits) if digits > 0 else QtWidgets.QSpinBox()
        spinbox.setMaximum(2147483647)
        spinbox.setMinimum(-2147483648)
        spinbox.setSuffix(f' {units}' if units else '')
        spinbox.setValue(default)
        hbox.addWidget(spinbox)
        self.layout.addLayout(hbox)
        self.controls.append((switch, spinbox, checkbox if checked in (True, False) else None))

    def initUI(self):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        horiz_layout = QtWidgets.QHBoxLayout()
        self.layout = QtWidgets.QVBoxLayout()
        horiz_layout.addLayout(self.layout)
        self.controls = []

        self.appendControl('Radius', 5, 'R', 'mm', digits=1)
        self.appendControl('Dipole', 0.2, 'dipole', 'T', False)
        self.appendControl('Quadrupole', 300, 'quad', 'T/m', True)
        self.appendControl('Sextupole', 66, 'sext', 'T/m²', False)
        self.appendControl('Octupole', 888, 'oct', 'T/m³', False)
        self.appendControl('Remanent field', 1.07, 'Br', 'T')
        self.appendControl('Wedges', 16, 'wedges', digits=0)
        self.appendControl('Symmetry', 1, 'symmetry', digits=0)

        hbox = QtWidgets.QHBoxLayout()
        self.checkbox = QtWidgets.QCheckBox('Auto')
        # self.checkbox.clicked.connect(self.run_simulation)
        hbox.addWidget(self.checkbox)
        run_button = QtWidgets.QPushButton('Run')
        run_button.clicked.connect(self.run_simulation)
        hbox.addWidget(run_button)
        self.layout.addLayout(hbox)

        self.plot = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.plot, self)  # TODO: make it appear above the plot
        horiz_layout.addWidget(self.plot)

        self.setLayout(horiz_layout)
        self.show()

    @pyqtSlot()
    def run_simulation(self):
        cmd = [halbach_exe, ]
        for switch, spinbox, checkbox in self.controls:
            if checkbox is None or checkbox.isChecked():
                value = spinbox.value() * (0.001 if switch == 'R' else 1)
                cmd.append(f'{switch}={value}')

        # if self.checkbox.isChecked():
        for line in execute(cmd):
            print(line, end="")
            # process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            # output, err = process.communicate()
            # exit_code = process.wait()
            # print(output)
        polygons = np.loadtxt(r"magnet.csv", skiprows=1, usecols=range(1, 11), delimiter=',')
        min_width = min(np.ptp(polygons[:, 2::2], axis=1))
        min_height = min(np.ptp(polygons[:, 3::2], axis=1))
        max_b = max(polygons[:, :2].flat)
        scale = min(min_width, min_height) / max_b
        print(scale)
        self.plot.axes.cla()
        shapes = [Polygon(polygon[2:10].reshape(4, 2)) for polygon in polygons]
        self.plot.axes.add_collection(PatchCollection(shapes))
        for polygon in polygons:
        #     print(polygon[[2, 4, 6, 8, 2]], polygon[[3, 5, 7, 9, 3]])
        #     self.plot.axes.plot(polygon[[2, 4, 6, 8, 2]], polygon[[3, 5, 7, 9, 3]], 'r')
            bx = polygon[0] * scale
            by = polygon[1] * scale
            self.plot.axes.arrow(np.mean(polygon[2::2]) - bx / 2, np.mean(polygon[3::2]) - by / 2,
                             bx, by, width=0.1*np.sqrt(bx**2 + by**2), length_includes_head=True)
            # self.plot.axes.plot(range(4), random.sample(range(10), 4))
        self.plot.axes.set_xlim([min(polygons[:, 2::2].flat), max(polygons[:, 2::2].flat)])
        self.plot.axes.set_ylim([min(polygons[:, 3::2].flat), max(polygons[:, 3::2].flat)])
        self.plot.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())